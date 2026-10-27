"""
11A_train_federated_convnext.py
--------------------------------

PHASE 2 - STEP 11A

CONVNEXT-TINY V2 FEDERATED TRAINING

This is the actual federated training driver for the ConvNeXt-Tiny V2
experiment 11A-v4 with client-local augmentation, class-balanced
sampling, smoothed/capped class weights, and fully automatic
client learning-rate adaptation).

Federated algorithm:
    Sample-weighted FedAvg

Pipeline:
    1. Create a single global ConvNeXt-Tiny V2 model.
    2. Initialize it with ImageNet pretrained weights.
    3. Create client models from the global state.
    4. Train each client locally.
    5. Extract local model states.
    6. Aggregate local states using sample-weighted FedAvg.
    7. Update the global model.
    8. Evaluate the global model on validation data.
    9. Save the best global checkpoint using validation Macro-F1.
   10. Continue until maximum rounds or early stopping.
   11. Reload the best global checkpoint.
   12. Evaluate once on the held-out test set.

IMPORTANT:
    The centralized ConvNeXt model is NOT used as the initialization
    for the federated experiment.

    Federated training starts from the same ImageNet pretrained
    ConvNeXt-Tiny initialization as the centralized experiment.

Hardware:
    Designed for NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM.

Expected project structure:

    fl_project/
    |
    +-- configs/
    |   +-- config.py
    |
    +-- src/
    |   +-- 06_dataset.py
    |   +-- 07_model_convnext.py
    |   +-- 09A_federated_client.py
    |   +-- 10A_fedavg_server_convnext.py
    |   +-- 11A_train_federated_convnext.py
    |
    +-- outputs/
        +-- models/
        +-- training_logs/
        +-- plots/

Outputs:

    outputs/models/federated_convnext_global_best.pt
    outputs/models/federated_convnext_global_final.pt
    outputs/training_logs/federated_convnext_training_log.csv
    outputs/training_logs/federated_convnext_test_classification_report.txt
    outputs/training_logs/federated_convnext_per_class_f1.csv
    outputs/plots/federated_convnext_training_curve.png
    outputs/plots/federated_convnext_confusion_matrix.png
"""

# =====================================================================
# IMPORTS
# =====================================================================

import os
import sys
import time
import copy
import random
import importlib.util

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler


# =====================================================================
# PATH SETUP
# =====================================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")

if CONFIG_DIR not in sys.path:
    sys.path.append(CONFIG_DIR)

import config


# =====================================================================
# MODULE LOADER
# =====================================================================

def _load_module(filename, module_name):
    """
    Dynamically load a Python module from the src directory.
    """

    path = os.path.join(SRC_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired module not found:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module:\n{path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# =====================================================================
# LOAD PROJECT MODULES
# =====================================================================

dataset_module = _load_module(
    "06_dataset.py",
    "dataset_module_step11a",
)

model_module = _load_module(
    "07A_model_convnext.py",
    "model_module_step11a",
)


# =====================================================================
# MODEL CLASS DISCOVERY
# =====================================================================

def get_model_class(module):
    """
    Find the ConvNeXt hybrid classifier class.

    The function first checks common expected names and then searches
    for a class whose name contains both "convnext" and "classifier".
    """

    preferred_names = [
        "ConvNeXtHybridClassifier",
        "ConvNeXtHybrid",
        "ConvNeXtClassifier",
        "HybridClassifier",
    ]

    for name in preferred_names:

        if hasattr(module, name):

            candidate = getattr(module, name)

            if isinstance(candidate, type):
                return candidate

    candidates = []

    for name in dir(module):

        if name.startswith("_"):
            continue

        candidate = getattr(module, name)

        if isinstance(candidate, type):

            lower_name = name.lower()

            if (
                "convnext" in lower_name
                and "classifier" in lower_name
            ):
                candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]

    raise AttributeError(
        "\nCould not automatically locate the ConvNeXt classifier "
        "inside 07_model_convnext.py.\n\n"
        "Expected one of:\n"
        "  ConvNeXtHybridClassifier\n"
        "  ConvNeXtHybrid\n"
        "  ConvNeXtClassifier\n"
        "  HybridClassifier\n"
    )


ConvNeXtModel = get_model_class(model_module)


# =====================================================================
# DATA LOADER DISCOVERY
# =====================================================================

def get_client_loader_function(module):
    """
    Locate the client-training DataLoader function.

    Supports the common naming used in the federated client module.
    """

    preferred_names = [
        "build_client_train_loader",
        "build_federated_client_loader",
        "build_client_loader",
    ]

    for name in preferred_names:

        if hasattr(module, name):

            candidate = getattr(module, name)

            if callable(candidate):
                return candidate

    raise AttributeError(
        "\nCould not find the client DataLoader function in "
        "06_dataset.py.\n\n"
        "Expected one of:\n"
        "  build_client_train_loader\n"
        "  build_federated_client_loader\n"
        "  build_client_loader\n"
    )


build_client_train_loader = get_client_loader_function(
    dataset_module
)


# =====================================================================
# EVALUATION LOADER
# =====================================================================

if not hasattr(dataset_module, "build_eval_loader"):

    raise AttributeError(
        "\n06_dataset.py does not contain build_eval_loader()."
    )

build_eval_loader = dataset_module.build_eval_loader


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_random_seeds(seed):
    """
    Make training as reproducible as practical.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings improve reproducibility.
    # Benchmarking is disabled because image dimensions are fixed but
    # deterministic behavior is preferred for this experiment.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed: {seed}")


# =====================================================================
# DEVICE
# =====================================================================

def get_device():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU VRAM: {total_memory:.2f} GB"
        )

    return device


# =====================================================================
# MODEL CONSTRUCTION
# =====================================================================

def build_global_model(device):
    """
    Build a fresh ImageNet-pretrained ConvNeXt-Tiny V2 model.

    This is deliberately independent of the centralized checkpoint.
    """

    print("\n" + "=" * 60)
    print("BUILDING GLOBAL CONVNEXT-TINY V2 MODEL")
    print("=" * 60)

    model = ConvNeXtModel(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=config.HANDCRAFTED_FEATURE_DIM,
        pretrained=True,
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nGlobal model:")
    print(
        f"  Total parameters    : {total_params:,}"
    )
    print(
        f"  Trainable parameters: {trainable_params:,}"
    )

    return model


# =====================================================================
# CLASS-BALANCED LOCAL LOSS
# =====================================================================

def compute_client_class_weights(client_loader, device):
    if not hasattr(client_loader, "dataset") or not hasattr(client_loader.dataset, "df"):
        raise AttributeError(
            "Client DataLoader dataset must expose a pandas DataFrame "
            "as dataset.df for client-local class weighting."
        )

    client_df = client_loader.dataset.df
    counts = client_df["label"].value_counts()
    total = len(client_df)
    num_classes = config.NUM_CLASSES

    if total <= 0:
        raise ValueError("Client contains zero samples.")

    present_classes = sum(
        1 for cls in config.CLASS_LIST
        if int(counts.get(cls, 0)) > 0
    )

    if present_classes == 0:
        raise ValueError("No known classes were found in the client dataset.")

    weight_power = float(getattr(config, "FED_CLASS_WEIGHT_POWER", 0.5))
    max_class_weight = float(getattr(config, "FED_MAX_CLASS_WEIGHT", 5.0))
    min_class_weight = float(getattr(config, "FED_MIN_CLASS_WEIGHT", 0.25))

    if not (0.0 < weight_power <= 1.0):
        raise ValueError("FED_CLASS_WEIGHT_POWER must be in (0, 1].")
    if max_class_weight <= 0:
        raise ValueError("FED_MAX_CLASS_WEIGHT must be > 0.")
    if min_class_weight <= 0:
        raise ValueError("FED_MIN_CLASS_WEIGHT must be > 0.")
    if min_class_weight > max_class_weight:
        raise ValueError("FED_MIN_CLASS_WEIGHT must be <= FED_MAX_CLASS_WEIGHT.")

    raw_weights = []
    weights = []

    print("\nClient-local class counts:")
    for cls in config.CLASS_LIST:
        count = int(counts.get(cls, 0))
        print(f"  {cls:6s}: {count}")

    for cls in config.CLASS_LIST:
        count = int(counts.get(cls, 0))

        if count == 0:
            raw_weight = 1.0
            weight = 1.0
        else:
            raw_weight = total / (present_classes * count)
            weight = raw_weight ** weight_power
            weight = max(min_class_weight, min(max_class_weight, weight))

        raw_weights.append(float(raw_weight))
        weights.append(float(weight))

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    print("\nClient-local raw class weights:")
    for cls, weight in zip(config.CLASS_LIST, raw_weights):
        print(f"  {cls:6s}: {weight:.4f}")

    print("\nClient-local smoothed/capped class weights:")
    for cls, weight in zip(config.CLASS_LIST, weights):
        print(f"  {cls:6s}: {weight:.4f}")

    print(
        f"\nClass-weight smoothing: power={weight_power:.2f}, "
        f"min={min_class_weight:.2f}, max={max_class_weight:.2f}."
    )

    return weights_tensor


def build_criterion(device, class_weights=None):
    """Build CrossEntropyLoss with optional class weighting."""
    label_smoothing = getattr(
        config,
        "FED_LABEL_SMOOTHING",
        getattr(config, "LABEL_SMOOTHING", 0.02),
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )

    if class_weights is None:
        print(
            f"\nUsing standard CrossEntropyLoss for evaluation "
            f"(label smoothing={label_smoothing})."
        )
    else:
        print(
            f"\nUsing client-local class-balanced "
            f"CrossEntropyLoss (label smoothing={label_smoothing})."
        )

    return criterion


# =====================================================================
# CLIENT TRAINING
# =====================================================================

def train_client(
    global_state,
    client_id,
    client_loader,
    device,
    criterion,
    use_amp,
    local_epochs,
    learning_rate,
    weight_decay,
    min_learning_rate=1e-7,
    max_learning_rate=None,
    lr_adaptation_rate=1.0,
):
    """
    Train one client from the current global model.

    Client-adaptive persistent learning-rate idea:

        local epoch accuracy increases
            -> decrease LR proportionally to the observed improvement

        local epoch accuracy is unchanged or decreases
            -> increase LR proportionally to the observed degradation/no-gain

    The controller is fully automatic: the multiplicative LR update is
    derived directly from the raw epoch-to-epoch accuracy delta using
    exp(-accuracy_delta). There is NO manually selected decay/recovery
    factor and NO accuracy threshold/dead-band.

    The final LR after the last local epoch is returned to the federated
    driver. Step 11A stores that value for this client and uses it as the
    starting LR when the SAME client participates in the next federated
    round.

    LR is optimizer/client state, so it is never averaged by FedAvg.
    """

    if local_epochs < 1:
        raise ValueError("local_epochs must be >= 1.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")

    if min_learning_rate <= 0:
        raise ValueError("min_learning_rate must be > 0.")

    if max_learning_rate is None:
        max_learning_rate = learning_rate * 2.0

    if lr_adaptation_rate <= 0:
        raise ValueError("lr_adaptation_rate must be > 0.")

    if max_learning_rate < min_learning_rate:
        raise ValueError(
            "max_learning_rate must be >= min_learning_rate."
        )

    # ---------------------------------------------------------------
    # LOCAL MODEL
    # ---------------------------------------------------------------

    local_model = ConvNeXtModel(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=config.HANDCRAFTED_FEATURE_DIM,
        pretrained=False,
    ).to(device)

    local_model.load_state_dict(
        global_state,
        strict=True,
    )

    local_model.train()

    # ---------------------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        local_model.parameters(),
        lr=float(learning_rate),
        weight_decay=weight_decay,
    )

    # ---------------------------------------------------------------
    # AMP
    # ---------------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # ---------------------------------------------------------------
    # LOCAL ADAPTIVE LR
    # ---------------------------------------------------------------

    previous_epoch_accuracy = None
    history = []

    final_loss = 0.0
    final_accuracy = 0.0
    final_sample_count = 0

    for local_epoch in range(1, local_epochs + 1):

        epoch_start_lr = optimizer.param_groups[0]["lr"]

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        progress = tqdm(
            client_loader,
            desc=(
                f"    Client {client_id} "
                f"epoch {local_epoch}/{local_epochs}"
            ),
            leave=False,
        )

        for images, features, labels in progress:

            images = images.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=True,
                ):
                    logits = local_model(images, features)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    local_model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()

            else:

                logits = local_model(images, features)
                loss = criterion(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    local_model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            batch_size = images.size(0)

            epoch_loss += loss.item() * batch_size

            predictions = torch.argmax(logits, dim=1)

            epoch_correct += (
                predictions == labels
            ).sum().item()

            epoch_total += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=(
                    f"{epoch_correct / epoch_total:.4f}"
                    if epoch_total > 0
                    else "0.0000"
                ),
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        if epoch_total == 0:
            raise ValueError(
                f"Client {client_id} contains zero samples."
            )

        epoch_loss /= epoch_total
        epoch_accuracy = epoch_correct / epoch_total

        # -----------------------------------------------------------
        # CORRECTED ADAPTIVE LR AFTER EVERY LOCAL EPOCH
        # -----------------------------------------------------------
        # First epoch establishes the accuracy baseline.
        # From epoch 2 onward:
        #
        #   accuracy increases
        #       -> decrease LR (fine-tune/converge)
        #
        #   accuracy is unchanged or decreases
        #       -> increase LR (recover/explore)
        #
        # This is intentionally based only on LOCAL EPOCH ACCURACY.
        # Validation/test data never controls the client optimizer.

        if previous_epoch_accuracy is None:
            lr_action = "initial"
            accuracy_delta = 0.0
            multiplier = 1.0
        else:
            accuracy_delta = (
                epoch_accuracy - previous_epoch_accuracy
            )

            # Fully automatic controller:
            #   delta > 0  -> exp(-delta) < 1  -> lower LR
            #   delta = 0  -> exp(0)    = 1    -> keep LR
            #   delta < 0  -> exp(-delta) > 1  -> raise LR
            #
            # The magnitude of the LR change is determined by how much
            # the local accuracy actually changed. No fixed .70/.1.10
            # multipliers are required.
            multiplier = float(np.exp(-lr_adaptation_rate * accuracy_delta))

            if accuracy_delta > 0.0:
                lr_action = "automatic_decrease_from_accuracy_gain"
            elif accuracy_delta < 0.0:
                lr_action = "automatic_increase_from_accuracy_drop"
            else:
                lr_action = "automatic_hold_no_accuracy_change"

        current_lr = optimizer.param_groups[0]["lr"]
        next_lr = current_lr * multiplier

        next_lr = max(
            min_learning_rate,
            min(max_learning_rate, next_lr),
        )

        optimizer.param_groups[0]["lr"] = next_lr

        print(
            f"      Client {client_id} local epoch {local_epoch}: "
            f"loss={epoch_loss:.4f}, "
            f"acc={epoch_accuracy:.4f}, "
            f"lr_start={epoch_start_lr:.2e}, "
            f"lr_end={next_lr:.2e}, "
            f"action={lr_action}"
        )

        history.append({
            "local_epoch": local_epoch,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_accuracy,
            "accuracy_delta": accuracy_delta,
            "learning_rate_start": epoch_start_lr,
            "learning_rate_end": next_lr,
            "lr_action": lr_action,
        })

        previous_epoch_accuracy = epoch_accuracy
        final_loss = epoch_loss
        final_accuracy = epoch_accuracy
        final_sample_count = epoch_total

    final_learning_rate = optimizer.param_groups[0]["lr"]

    local_state = {
        key: value.detach().cpu().clone()
        for key, value in local_model.state_dict().items()
    }

    del local_model
    del optimizer
    del scaler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        local_state,
        final_sample_count,
        final_loss,
        final_accuracy,
        final_learning_rate,
        history,
    )


# =====================================================================
# FEDAVG
# =====================================================================

def fedavg(client_states, client_sample_counts):
    """
    Standard sample-weighted FedAvg.

    For floating-point tensors:

        W_global =
            sum(n_k / N * W_k)

    where:
        n_k = samples at client k
        N   = total samples
    """

    if len(client_states) == 0:
        raise ValueError(
            "No client states supplied to FedAvg."
        )

    if len(client_states) != len(
        client_sample_counts
    ):

        raise ValueError(
            "Number of client states and "
            "sample counts do not match."
        )

    total_samples = sum(
        client_sample_counts
    )

    if total_samples <= 0:
        raise ValueError(
            "Total client samples must be > 0."
        )

    # ---------------------------------------------------------------
    # WEIGHTS
    # ---------------------------------------------------------------

    weights = [
        count / total_samples
        for count in client_sample_counts
    ]

    # ---------------------------------------------------------------
    # INITIALIZE FROM FIRST CLIENT
    # ---------------------------------------------------------------

    aggregated = {}

    reference_state = client_states[0]

    for key, tensor in reference_state.items():

        # Floating point parameters are averaged.
        if torch.is_floating_point(tensor):

            result = torch.zeros_like(
                tensor,
                dtype=torch.float32,
            )

            for state, weight in zip(
                client_states,
                weights,
            ):

                result += (
                    state[key].float()
                    * weight
                )

            aggregated[key] = result

        else:

            # Integer buffers such as counters are not mathematically
            # averaged. Keep the first client's value.
            aggregated[key] = tensor.clone()

    return aggregated


# =====================================================================
# STATE VALIDATION
# =====================================================================

def validate_state_dict(
    state_dict,
    reference_state,
):
    """
    Verify that a client state is structurally compatible with the
    global model.
    """

    if set(state_dict.keys()) != set(
        reference_state.keys()
    ):

        missing = (
            set(reference_state.keys())
            - set(state_dict.keys())
        )

        extra = (
            set(state_dict.keys())
            - set(reference_state.keys())
        )

        raise ValueError(
            "\nInvalid federated state dictionary."
            f"\nMissing keys: {list(missing)[:10]}"
            f"\nExtra keys: {list(extra)[:10]}"
        )

    for key in reference_state.keys():

        if state_dict[key].shape != (
            reference_state[key].shape
        ):

            raise ValueError(
                f"Shape mismatch for '{key}': "
                f"{state_dict[key].shape} vs "
                f"{reference_state[key].shape}"
            )


# =====================================================================
# GLOBAL EVALUATION
# =====================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion,
):
    """
    Evaluate validation or test data.
    """

    model.eval()

    total_loss = 0.0

    all_preds = []
    all_labels = []

    for images, features, labels in tqdm(
        loader,
        desc="  evaluate",
        leave=False,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images,
            features,
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_preds.extend(
            predictions.cpu().numpy().tolist()
        )

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

    if len(loader.dataset) == 0:

        raise ValueError(
            "Evaluation loader contains zero samples."
        )

    average_loss = (
        total_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds,
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        all_preds,
        all_labels,
    )


# =====================================================================
# PER-CLASS F1
# =====================================================================

def calculate_per_class_f1(
    labels,
    predictions,
):
    """
    Calculate F1 independently for every class.
    """

    scores = f1_score(
        labels,
        predictions,
        labels=list(
            range(config.NUM_CLASSES)
        ),
        average=None,
        zero_division=0,
    )

    return {
        cls: float(score)
        for cls, score
        in zip(
            config.CLASS_LIST,
            scores,
        )
    }


# =====================================================================
# CONFUSION MATRIX
# =====================================================================

def plot_confusion_matrix(
    labels,
    predictions,
    save_path,
):
    """
    Save confusion matrix for the held-out test set.
    """

    class_indices = list(
        range(config.NUM_CLASSES)
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=class_indices,
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(cm)

    ax.set_xticks(
        class_indices
    )

    ax.set_yticks(
        class_indices
    )

    ax.set_xticklabels(
        config.CLASS_LIST
    )

    ax.set_yticklabels(
        config.CLASS_LIST
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_title(
        "Federated ConvNeXt-Tiny V2 - Test Confusion Matrix"
    )

    threshold = (
        cm.max() / 2
        if cm.size > 0
        else 0
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            text_color = (
                "white"
                if cm[i, j] > threshold
                else "black"
            )

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved confusion matrix:\n  {save_path}"
    )


# =====================================================================
# TRAINING CURVES
# =====================================================================

def plot_training_curves(
    log_df,
    save_path,
):
    """
    Save federated training curves.
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 5),
    )

    # ---------------------------------------------------------------
    # LOSS
    # ---------------------------------------------------------------

    axes[0].plot(
        log_df["round"],
        log_df["val_loss"],
        label="Validation Loss",
        marker="o",
    )

    axes[0].set_xlabel(
        "Federated Round"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].set_title(
        "Federated ConvNeXt - Validation Loss"
    )

    axes[0].legend()
    axes[0].grid(
        alpha=0.3
    )

    # ---------------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------------

    axes[1].plot(
        log_df["round"],
        log_df["val_acc"],
        label="Validation Accuracy",
        marker="o",
    )

    axes[1].plot(
        log_df["round"],
        log_df["val_macro_f1"],
        label="Validation Macro-F1",
        marker="s",
    )

    axes[1].set_xlabel(
        "Federated Round"
    )

    axes[1].set_ylabel(
        "Score"
    )

    axes[1].set_title(
        "Federated ConvNeXt - Validation Metrics"
    )

    axes[1].legend()
    axes[1].grid(
        alpha=0.3
    )

    # ---------------------------------------------------------------
    # CLIENT TRAINING
    # ---------------------------------------------------------------

    axes[2].plot(
        log_df["round"],
        log_df["mean_client_loss"],
        label="Mean Client Loss",
        marker="o",
    )

    axes[2].plot(
        log_df["round"],
        log_df["mean_client_acc"],
        label="Mean Client Accuracy",
        marker="s",
    )

    axes[2].set_xlabel(
        "Federated Round"
    )

    axes[2].set_ylabel(
        "Value"
    )

    axes[2].set_title(
        "Federated ConvNeXt - Client Training"
    )

    axes[2].legend()
    axes[2].grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved training curves:\n  {save_path}"
    )


# =====================================================================
# CHECKPOINT SAVING
# =====================================================================

def save_checkpoint(
    path,
    model,
    round_number,
    val_loss,
    val_acc,
    val_macro_f1,
    best_val_f1,
    config_snapshot,
    client_learning_rates=None,
):
    """
    Save a complete global federated checkpoint.
    """

    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        },

        "round": round_number,

        "val_loss": val_loss,

        "val_acc": val_acc,

        "val_macro_f1": val_macro_f1,

        "best_val_f1": best_val_f1,

        "class_list": config.CLASS_LIST,

        "num_classes": config.NUM_CLASSES,

        "handcrafted_feature_dim":
            config.HANDCRAFTED_FEATURE_DIM,

        "random_seed":
            config.RANDOM_SEED,

        "config": config_snapshot,

        # Client optimizer state is saved separately from model weights.
        # It is never averaged by FedAvg.
        "client_learning_rates": {
            int(k): float(v)
            for k, v in (client_learning_rates or {}).items()
        },
    }

    torch.save(
        checkpoint,
        path,
    )


# =====================================================================
# CLIENT ID DISCOVERY
# =====================================================================

def get_client_ids():
    """
    Determine client IDs from config.

    Supported configurations:
        NUM_CLIENTS
        NUM_FEDERATED_CLIENTS

    If neither exists, default to 10 clients.
    """

    if hasattr(
        config,
        "NUM_CLIENTS",
    ):

        num_clients = int(
            config.NUM_CLIENTS
        )

    elif hasattr(
        config,
        "NUM_FEDERATED_CLIENTS",
    ):

        num_clients = int(
            config.NUM_FEDERATED_CLIENTS
        )

    else:

        num_clients = 10

    if num_clients <= 0:

        raise ValueError(
            "Number of clients must be > 0."
        )

    return list(
        range(num_clients)
    )


# =====================================================================
# CLIENT SAMPLING
# =====================================================================

def select_clients(
    client_ids,
    round_number,
    clients_per_round,
):
    """
    Deterministically select clients for each round.

    If clients_per_round >= number of clients, all clients
    participate.

    Otherwise a reproducible subset is selected.
    """

    if clients_per_round >= len(
        client_ids
    ):

        return list(client_ids)

    rng = random.Random(
        config.RANDOM_SEED
        + round_number
    )

    selected = rng.sample(
        client_ids,
        clients_per_round,
    )

    return sorted(
        selected
    )


# =====================================================================
# CLIENT LOADER
# =====================================================================

def build_client_loader(client_id):
    """
    Build a client DataLoader with local class-balanced sampling.

    The underlying dataset comes directly from 06_dataset.py, so its
    existing on-the-fly training augmentation is retained.

    WeightedRandomSampler:
        - operates only on the local training subset
        - samples existing records with replacement
        - creates no image files and does not duplicate data on disk
        - keeps num_samples == len(dataset), so FedAvg still uses the
          original client sample count for aggregation
        - never affects validation or test loaders
    """
    try:
        base_loader = build_client_train_loader(
            client_id
        )
    except TypeError:
        base_loader = build_client_train_loader(
            client_id=client_id
        )

    dataset = base_loader.dataset

    if len(dataset) == 0:
        return base_loader

    if not hasattr(dataset, "df"):
        raise AttributeError(
            "Client dataset must expose dataset.df for local sampling."
        )

    client_df = dataset.df.reset_index(drop=True)

    if "label" not in client_df.columns:
        raise AttributeError(
            "Client dataset dataframe must contain a 'label' column."
        )

    # ---------------------------------------------------------------
    # TEMPERED / CAPPED SAMPLER WEIGHTS
    # ---------------------------------------------------------------
    #
    # Class sampling uses a square-root inverse-frequency rule:
    #
    #     raw_weight = N / (K_present * class_count)
    #     sampler_weight = raw_weight ** power
    #
    # The cap prevents clients with extremely rare classes from
    # repeatedly drawing the same few images too aggressively.
    # ---------------------------------------------------------------

    sampler_power = float(
        getattr(config, "FED_SAMPLER_WEIGHT_POWER", 0.5)
    )
    sampler_min_weight = float(
        getattr(config, "FED_MIN_SAMPLER_WEIGHT", 0.25)
    )
    sampler_max_weight = float(
        getattr(config, "FED_MAX_SAMPLER_WEIGHT", 5.0)
    )

    if not (0.0 < sampler_power <= 1.0):
        raise ValueError(
            "FED_SAMPLER_WEIGHT_POWER must be in (0, 1]."
        )

    if sampler_min_weight <= 0:
        raise ValueError(
            "FED_MIN_SAMPLER_WEIGHT must be > 0."
        )

    if sampler_max_weight < sampler_min_weight:
        raise ValueError(
            "FED_MAX_SAMPLER_WEIGHT must be >= FED_MIN_SAMPLER_WEIGHT."
        )

    counts = client_df["label"].value_counts()
    total = len(client_df)

    present_classes = [
        cls
        for cls in config.CLASS_LIST
        if int(counts.get(cls, 0)) > 0
    ]

    if not present_classes:
        raise ValueError(
            f"Client {client_id} contains no recognized classes."
        )

    class_to_sampler_weight = {}

    for cls in config.CLASS_LIST:
        count = int(counts.get(cls, 0))

        if count == 0:
            # Missing classes must not be sampled.
            class_to_sampler_weight[cls] = 0.0
            continue

        raw_weight = total / (len(present_classes) * count)

        weight = raw_weight ** sampler_power

        weight = max(
            sampler_min_weight,
            min(sampler_max_weight, weight)
        )

        class_to_sampler_weight[cls] = float(weight)

    sample_weights = torch.tensor(
        [
            class_to_sampler_weight[label]
            for label in client_df["label"].tolist()
        ],
        dtype=torch.double,
    )

    if not torch.isfinite(sample_weights).all():
        raise ValueError(
            f"Client {client_id}: sampler produced NaN/Inf weights."
        )

    if not torch.any(sample_weights > 0):
        raise ValueError(
            f"Client {client_id}: sampler has no positive weights."
        )

    # One epoch contains exactly the original number of client samples.
    # Replacement allows minority examples to be seen repeatedly.
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )

    balanced_loader = DataLoader(
        dataset,
        batch_size=base_loader.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # ---------------------------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------------------------
    print(
        f"\nClient {client_id} local class-balanced sampler:"
    )
    print(
        f"  Sampler draws/epoch : {len(dataset)}"
    )
    print(
        f"  Sampling power      : {sampler_power:.2f}"
    )
    print(
        f"  Sampling weight cap : "
        f"[{sampler_min_weight:.2f}, {sampler_max_weight:.2f}]"
    )

    print("  Per-class sampler weights:")
    for cls in config.CLASS_LIST:
        print(
            f"    {cls:6s}: "
            f"count={int(counts.get(cls, 0)):4d}, "
            f"weight={class_to_sampler_weight[cls]:.4f}"
        )

    print("  ✓ WeightedRandomSampler enabled (replacement=True)")
    print("  ✓ Existing on-the-fly training augmentation retained")
    print("  ✓ Original client sample count preserved for FedAvg")

    return balanced_loader


# =====================================================================
# MAIN
# =====================================================================

def main():

    start_time = time.perf_counter()

    print("=" * 60)
    print(
        "PHASE 2 - STEP 11A-v4: "
        "CONVNEXT-TINY V2 FEDERATED TRAINING"
    )
    print("=" * 60)

    # =================================================================
    # DEVICE
    # =================================================================

    device = get_device()

    # =================================================================
    # SEED
    # =================================================================

    seed = getattr(
        config,
        "RANDOM_SEED",
        42,
    )

    set_random_seeds(
        seed
    )

    # =================================================================
    # CONFIGURATION
    # =================================================================

    local_epochs = getattr(
        config,
        "FED_LOCAL_EPOCHS",
        getattr(
            config,
            "LOCAL_EPOCHS",
            3,
        ),
    )

    num_rounds = getattr(
        config,
        "FED_NUM_ROUNDS",
        getattr(
            config,
            "NUM_FEDERATED_ROUNDS",
            30,
        ),
    )

    clients_per_round = getattr(
        config,
        "FED_CLIENTS_PER_ROUND",
        getattr(
            config,
            "CLIENTS_PER_ROUND",
            None,
        ),
    )

    if clients_per_round is None:
        clients_per_round = getattr(
            config,
            "NUM_CLIENTS",
            getattr(
                config,
                "NUM_FEDERATED_CLIENTS",
                10,
            ),
        )

    learning_rate = getattr(
        config,
        "FED_LEARNING_RATE",
        getattr(
            config,
            "LEARNING_RATE",
            getattr(
                config,
                "LOCAL_LEARNING_RATE",
                5e-5,
            ),
        ),
    )

    weight_decay = getattr(
        config,
        "FED_WEIGHT_DECAY",
        getattr(
            config,
            "WEIGHT_DECAY",
            1e-4,
        ),
    )

    patience = getattr(
        config,
        "FED_EARLY_STOPPING_PATIENCE",
        getattr(
            config,
            "EARLY_STOPPING_PATIENCE",
            8,
        ),
    )

    min_delta = getattr(
        config,
        "FED_MIN_DELTA",
        1e-4,
    )

    use_amp = (
        getattr(
            config,
            "USE_AMP",
            True,
        )
        and torch.cuda.is_available()
    )

    grad_clip = 1.0

    print("\n" + "=" * 60)
    print(
        "FEDERATED TRAINING CONFIGURATION"
    )
    print("=" * 60)

    print(
        "\nArchitecture          : "
        "ConvNeXt-Tiny V2"
    )

    print(
        f"Device                : {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU                   : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Federated rounds      : {num_rounds}"
    )

    print(
        f"Local epochs          : {local_epochs}"
    )

    print(
        f"Clients per round     : "
        f"{clients_per_round}"
    )

    print(
        f"Local learning rate   : "
        f"{learning_rate}"
    )

    print(
        f"Weight decay          : "
        f"{weight_decay}"
    )

    print(
        f"Label smoothing      : "
        f"{getattr(config, 'FED_LABEL_SMOOTHING', getattr(config, 'LABEL_SMOOTHING', 0.02))}"
    )

    print(
        f"AMP                   : {use_amp}"
    )

    print(
        f"Gradient clipping     : "
        f"{grad_clip}"
    )

    print(
        "Local sampling        : WeightedRandomSampler"
    )
    print(
        f"Sampler power         : "
        f"{float(getattr(config, 'FED_SAMPLER_WEIGHT_POWER', 0.5)):.2f}"
    )
    print(
        f"Sampler weight cap    : "
        f"[{float(getattr(config, 'FED_MIN_SAMPLER_WEIGHT', 0.25)):.2f}, "
        f"{float(getattr(config, 'FED_MAX_SAMPLER_WEIGHT', 5.0)):.2f}]"
    )

    print(
        f"Early stopping        : "
        f"{patience} rounds"
    )

    print(
        f"Classes               : "
        f"{config.NUM_CLASSES}"
    )

    print(
        f"Handcrafted features  : "
        f"{config.HANDCRAFTED_FEATURE_DIM}"
    )

    # =================================================================
    # CLIENTS
    # =================================================================

    client_ids = get_client_ids()

    print(
        f"\nTotal clients configured: "
        f"{len(client_ids)}"
    )

    print(
        f"Client IDs: {client_ids}"
    )

    # =================================================================
    # OUTPUT DIRECTORIES
    # =================================================================

    model_dir = os.path.join(
        config.OUTPUT_DIR,
        "models",
    )

    log_dir = os.path.join(
        config.OUTPUT_DIR,
        "training_logs",
    )

    plots_dir = os.path.join(
        config.OUTPUT_DIR,
        "plots",
    )

    os.makedirs(
        model_dir,
        exist_ok=True,
    )

    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    os.makedirs(
        plots_dir,
        exist_ok=True,
    )

    experiment_tag = getattr(
        config,
        "FED_EXPERIMENT_TAG",
        "11A_v4_aug_sampler_class_balanced",
    )

    best_model_path = os.path.join(
        model_dir,
        f"federated_convnext_{experiment_tag}_global_best.pt",
    )

    final_model_path = os.path.join(
        model_dir,
        f"federated_convnext_{experiment_tag}_global_final.pt",
    )

    log_path = os.path.join(
        log_dir,
        f"federated_convnext_{experiment_tag}_training_log.csv",
    )

    # =================================================================
    # BUILD GLOBAL MODEL
    # =================================================================

    global_model = build_global_model(
        device
    )

    # =================================================================
    # EVALUATION LOSS
    # =================================================================

    print("\n" + "=" * 60)
    print("BUILDING GLOBAL EVALUATION LOSS")
    print("=" * 60)

    # Evaluation uses the ordinary (unweighted) loss.  Class balancing is
    # applied ONLY inside each client's local training objective.
    eval_criterion = build_criterion(
        device,
        class_weights=None,
    )

    # =================================================================
    # VALIDATION LOADER
    # =================================================================

    print("\n" + "=" * 60)
    print("BUILDING VALIDATION DATALOADER")
    print("=" * 60)

    val_loader = build_eval_loader(
        "val"
    )

    print(
        f"\nValidation samples: "
        f"{len(val_loader.dataset)}"
    )

    # =================================================================
    # TEST LOADER
    # =================================================================

    # Built only for final evaluation.
    print("\n" + "=" * 60)
    print("BUILDING TEST DATALOADER")
    print("=" * 60)

    test_loader = build_eval_loader(
        "test"
    )

    print(
        f"\nTest samples: "
        f"{len(test_loader.dataset)}"
    )

    # =================================================================
    # INITIAL GLOBAL STATE
    # =================================================================

    global_state = {
        key: value.detach().cpu().clone()
        for key, value
        in global_model.state_dict().items()
    }

    print(
        "\nInitial global state prepared."
    )

    # =================================================================
    # CLIENT-PERSISTENT AUTOMATIC ADAPTIVE LEARNING RATE
    # =================================================================
    #
    # Each client has its own LR. It is initialized once from the base
    # learning rate and is then carried across federated rounds.
    #
    # IMPORTANT:
    #   Client LR is NOT a model parameter and is NOT averaged by FedAvg.
    #   Only model weights are sent to the server for aggregation.

    client_learning_rates = {
        int(client_id): float(learning_rate)
        for client_id in client_ids
    }

    # Fully automatic LR adaptation.
    # The controller derives its update from each local epoch's
    # accuracy change, so no fixed decay/recovery factors are used.
    min_learning_rate = float(
        getattr(
            config,
            "FED_MIN_LEARNING_RATE",
            max(learning_rate * 0.05, 1e-7),
        )
    )

    max_learning_rate = float(
        getattr(
            config,
            "FED_MAX_LEARNING_RATE",
            learning_rate * 2.0,
        )
    )

    lr_adaptation_rate = float(
        getattr(
            config,
            "FED_LR_ADAPTATION_RATE",
            1.0,
        )
    )

    if min_learning_rate <= 0:
        raise ValueError(
            "FED_MIN_LEARNING_RATE must be > 0."
        )

    if max_learning_rate < min_learning_rate:
        raise ValueError(
            "FED_MAX_LEARNING_RATE must be >= FED_MIN_LEARNING_RATE."
        )

    if lr_adaptation_rate <= 0:
        raise ValueError(
            "FED_LR_ADAPTATION_RATE must be > 0."
        )

    print("\n" + "=" * 60)
    print("CLIENT-PERSISTENT ADAPTIVE LEARNING RATE")
    print("=" * 60)
    print(f"Base LR                 : {learning_rate:.2e}")
    print("LR adaptation           : automatic from accuracy delta")
    print("LR rule                 : next_lr = current_lr * exp(-adaptation_rate * accuracy_delta)")
    print(f"Minimum LR              : {min_learning_rate:.2e}")
    print(f"Maximum LR              : {max_learning_rate:.2e}")
    print("Accuracy threshold      : disabled")

    print("\nInitial client LR state:")
    for client_id in client_ids:
        print(
            f"  Client {client_id}: "
            f"{client_learning_rates[client_id]:.2e}"
        )

    # =================================================================
    # BEST MODEL TRACKING
    # =================================================================

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_round = 0

    rounds_without_improvement = 0

    log_rows = []

    # =================================================================
    # FEDERATED TRAINING
    # =================================================================

    print("\n" + "=" * 60)
    print(
        "STARTING FEDERATED CONVNEXT-TINY V2 TRAINING"
    )
    print("=" * 60)

    for round_number in range(
        1,
        num_rounds + 1,
    ):

        round_start = time.perf_counter()

        print("\n")
        print("=" * 60)
        print(
            f"FEDERATED ROUND "
            f"{round_number}/{num_rounds}"
        )
        print("=" * 60)

        # =============================================================
        # SELECT CLIENTS
        # =============================================================

        selected_clients = select_clients(
            client_ids,
            round_number,
            int(clients_per_round),
        )

        print(
            f"\nSelected clients: "
            f"{selected_clients}"
        )

        # =============================================================
        # CLIENT-SPECIFIC LR AT ROUND START
        # =============================================================

        print("\nClient learning rates entering this round:")

        for client_id in selected_clients:
            print(
                f"  Client {client_id}: "
                f"{client_learning_rates[client_id]:.2e}"
            )

        # =============================================================
        # CLIENT TRAINING
        # =============================================================

        client_states = []
        client_sample_counts = []

        client_losses = []
        client_accuracies = []
        client_histories = {}

        for client_id in selected_clients:

            print("\n" + "-" * 60)
            print(
                f"CLIENT {client_id}"
            )
            print("-" * 60)

            # ---------------------------------------------------------
            # BUILD CLIENT LOADER
            # ---------------------------------------------------------

            client_loader = build_client_loader(
                client_id
            )

            # FedAvg uses the TRUE local dataset size, not the number
            # of sampler draws. WeightedRandomSampler has replacement=True
            # but num_samples is exactly len(dataset).
            num_samples = len(
                client_loader.dataset
            )

            if num_samples <= 0:

                print(
                    f"WARNING: Client {client_id} "
                    f"contains zero samples. Skipping."
                )

                continue

            print(
                f"Client samples: "
                f"{num_samples}"
            )

            # ---------------------------------------------------------
            # TRAIN CLIENT
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            # BUILD CLIENT-LOCAL CLASS-BALANCED LOSS
            # ---------------------------------------------------------
            client_class_weights = compute_client_class_weights(
                client_loader,
                device,
            )

            local_criterion = build_criterion(
                device,
                class_weights=client_class_weights,
            )

            try:

                (
                    local_state,
                    sample_count,
                    client_loss,
                    client_acc,
                    final_client_lr,
                    client_history,
                ) = train_client(
                    global_state=global_state,
                    client_id=client_id,
                    client_loader=client_loader,
                    device=device,
                    criterion=local_criterion,
                    use_amp=use_amp,
                    local_epochs=int(
                        local_epochs
                    ),
                    learning_rate=float(
                        client_learning_rates[client_id]
                    ),
                    weight_decay=float(
                        weight_decay
                    ),
                    min_learning_rate=min_learning_rate,
                    max_learning_rate=max_learning_rate,
                    lr_adaptation_rate=lr_adaptation_rate,
                )

            except RuntimeError as exc:

                # Explicit CUDA OOM message so the user knows what
                # happened instead of silently continuing.
                if (
                    "out of memory"
                    in str(exc).lower()
                ):

                    print(
                        "\nCUDA OUT OF MEMORY "
                        f"on Client {client_id}."
                    )

                    print(
                        "Reduce the federated/client "
                        "batch size in config.py "
                        "from 16 to 8."
                    )

                    raise

                raise

            # ---------------------------------------------------------
            # VALIDATE STATE
            # ---------------------------------------------------------

            validate_state_dict(
                local_state,
                global_state,
            )

            # ---------------------------------------------------------
            # STORE CLIENT UPDATE
            # ---------------------------------------------------------

            client_states.append(
                local_state
            )

            client_sample_counts.append(
                sample_count
            )

            client_losses.append(
                client_loss
            )

            client_accuracies.append(
                client_acc
            )

            # =========================================================
            # PERSIST THIS CLIENT'S FINAL LR
            # =========================================================
            # Round 1 final LR -> Round 2 starting LR for this client.
            client_learning_rates[client_id] = float(
                final_client_lr
            )
            client_histories[client_id] = client_history

            print(
                f"\nClient {client_id} complete:"
            )

            print(
                f"  Samples : {sample_count}"
            )

            print(
                f"  Loss    : {client_loss:.4f}"
            )

            print(
                f"  Accuracy: {client_acc:.4f}"
            )

            print(
                f"  Final LR: {final_client_lr:.2e}"
            )

            print(
                "  Persisted for next round: "
                f"Client {client_id} -> "
                f"{client_learning_rates[client_id]:.2e}"
            )

            # ---------------------------------------------------------
            # CLEAN LOADER REFERENCE
            # ---------------------------------------------------------

            del client_loader

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # =============================================================
        # CHECK CLIENT UPDATES
        # =============================================================

        if len(client_states) == 0:

            raise RuntimeError(
                "No valid client updates were produced "
                f"in round {round_number}."
            )

        # =============================================================
        # FEDAVG AGGREGATION
        # =============================================================

        print("\n" + "-" * 60)
        print(
            "SERVER: SAMPLE-WEIGHTED FEDAVG"
        )
        print("-" * 60)

        total_client_samples = sum(
            client_sample_counts
        )

        print(
            f"\nTotal participating samples: "
            f"{total_client_samples}"
        )

        for i, (client_id, count) in enumerate(
            zip(
                selected_clients,
                client_sample_counts,
            )
        ):

            weight = (
                count /
                total_client_samples
            )

            print(
                f"  Client {client_id}: "
                f"samples={count}, "
                f"weight={weight:.6f}"
            )

        # -------------------------------------------------------------
        # AGGREGATE
        # -------------------------------------------------------------

        new_global_state = fedavg(
            client_states,
            client_sample_counts,
        )

        # -------------------------------------------------------------
        # VALIDATE GLOBAL STATE
        # -------------------------------------------------------------

        validate_state_dict(
            new_global_state,
            global_state,
        )

        # -------------------------------------------------------------
        # UPDATE GLOBAL STATE
        # -------------------------------------------------------------

        global_state = new_global_state

        global_model.load_state_dict(
            global_state,
            strict=True,
        )

        print(
            "\n✓ FedAvg aggregation complete."
        )

        # =============================================================
        # CLIENT LR STATE AFTER LOCAL TRAINING
        # =============================================================

        print("\nClient learning rates after this round:")
        for client_id in selected_clients:
            print(
                f"  Client {client_id}: "
                f"{client_learning_rates[client_id]:.2e}"
            )

        # =============================================================
        # GLOBAL VALIDATION
        # =============================================================

        print("\n" + "-" * 60)
        print(
            "SERVER: GLOBAL VALIDATION"
        )
        print("-" * 60)

        (
            val_loss,
            val_acc,
            val_macro_f1,
            _,
            _,
        ) = evaluate(
            model=global_model,
            loader=val_loader,
            device=device,
            criterion=eval_criterion,
        )

        # =============================================================
        # CLIENT SUMMARY
        # =============================================================

        mean_client_loss = float(
            np.mean(client_losses)
        )

        mean_client_acc = float(
            np.mean(client_accuracies)
        )

        round_client_lr_end = {
            int(client_id): float(client_learning_rates[client_id])
            for client_id in selected_clients
        }

        mean_client_lr_end = float(
            np.mean(list(round_client_lr_end.values()))
        )

        round_time = (
            time.perf_counter()
            - round_start
        )

        # =============================================================
        # PRINT ROUND RESULTS
        # =============================================================

        print("\n" + "=" * 60)
        print(
            f"ROUND {round_number} RESULTS"
        )
        print("=" * 60)

        print(
            f"  Participating clients : "
            f"{len(client_states)}"
        )

        print(
            f"  Client samples        : "
            f"{total_client_samples}"
        )

        print(
            f"  Mean client loss      : "
            f"{mean_client_loss:.4f}"
        )

        print(
            f"  Mean client accuracy  : "
            f"{mean_client_acc:.4f}"
        )

        print(
            f"  Validation loss       : "
            f"{val_loss:.4f}"
        )

        print(
            f"  Validation accuracy   : "
            f"{val_acc:.4f}"
        )

        print(
            f"  Validation Macro-F1   : "
            f"{val_macro_f1:.4f}"
        )

        print(
            f"  Mean client LR (end)  : "
            f"{mean_client_lr_end:.2e}"
        )

        print(
            f"  Round time            : "
            f"{round_time:.1f}s"
        )

        # =============================================================
        # LOGGING
        # =============================================================

        log_rows.append(
            {
                "round": round_number,

                "participating_clients":
                    len(client_states),

                "client_samples":
                    total_client_samples,

                "mean_client_loss":
                    mean_client_loss,

                "mean_client_acc":
                    mean_client_acc,

                "val_loss":
                    val_loss,

                "val_acc":
                    val_acc,

                "val_macro_f1":
                    val_macro_f1,

                "mean_client_learning_rate_end":
                    mean_client_lr_end,

                "lr_accuracy_threshold":
                    "disabled",

                "client_learning_rates":
                    str(round_client_lr_end),

                "local_class_balanced_loss":
                    True,

                "client_epoch_histories":
                    str(client_histories),

                "round_time_sec":
                    round_time,
            }
        )

        # =============================================================
        # BEST CHECKPOINT
        # =============================================================

        improvement = (
            val_macro_f1
            > best_val_f1 + min_delta
        )

        if improvement:

            best_val_f1 = val_macro_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_round = round_number

            rounds_without_improvement = 0

            config_snapshot = {
                "architecture":
                    "ConvNeXt-Tiny V2",
                "experiment":
                    experiment_tag,

                "local_epochs":
                    int(local_epochs),

                "num_rounds":
                    int(num_rounds),

                "clients_per_round":
                    int(clients_per_round),

                "learning_rate":
                    float(learning_rate),

                "weight_decay":
                    float(weight_decay),

                "use_amp":
                    bool(use_amp),

                "gradient_clip":
                    float(grad_clip),

                "label_smoothing":
                    float(
                        getattr(
                            config,
                            "FED_LABEL_SMOOTHING",
                            getattr(
                                config,
                                "LABEL_SMOOTHING",
                                0.02,
                            ),
                        )
                    ),

                "local_class_balanced_loss": True,
                "local_class_weighting": "smoothed_capped_inverse_frequency_per_client",
                "class_weight_power": float(getattr(config, "FED_CLASS_WEIGHT_POWER", 0.5)),
                "min_class_weight": float(getattr(config, "FED_MIN_CLASS_WEIGHT", 0.25)),
                "max_class_weight": float(getattr(config, "FED_MAX_CLASS_WEIGHT", 5.0)),

                "local_class_balanced_sampling": True,
                "sampling_method": "WeightedRandomSampler",
                "sampling_replacement": True,
                "sampler_weight_power": float(
                    getattr(config, "FED_SAMPLER_WEIGHT_POWER", 0.5)
                ),
                "sampler_min_weight": float(
                    getattr(config, "FED_MIN_SAMPLER_WEIGHT", 0.25)
                ),
                "sampler_max_weight": float(
                    getattr(config, "FED_MAX_SAMPLER_WEIGHT", 5.0)
                ),

                "algorithm":
                    "Sample-weighted FedAvg",

                "initialization":
                    "ImageNet pretrained",

                "client_adaptive_learning_rate":
                    True,

                "lr_adaptation":
                    "automatic_exp_accuracy_delta",

                "lr_adaptation_rate":
                    float(lr_adaptation_rate),

                "min_learning_rate":
                    float(min_learning_rate),

                "max_learning_rate":
                    float(max_learning_rate),

                "accuracy_threshold":
                    "disabled",

                "lr_update_rule":
                    "next_lr = current_lr * exp(-adaptation_rate * accuracy_delta)",
            }

            save_checkpoint(
                path=best_model_path,
                model=global_model,
                round_number=round_number,
                val_loss=val_loss,
                val_acc=val_acc,
                val_macro_f1=val_macro_f1,
                best_val_f1=best_val_f1,
                config_snapshot=config_snapshot,
                client_learning_rates=client_learning_rates,
            )

            print(
                "\n✓ NEW BEST GLOBAL MODEL"
            )

            print(
                f"  Best round      : "
                f"{best_round}"
            )

            print(
                f"  Val Macro-F1    : "
                f"{best_val_f1:.4f}"
            )

            print(
                f"  Val Accuracy    : "
                f"{best_val_acc:.4f}"
            )

            print(
                f"  Val Loss        : "
                f"{best_val_loss:.4f}"
            )

            print(
                f"  Saved to:\n"
                f"    {best_model_path}"
            )

        else:

            rounds_without_improvement += 1

            print(
                "\nNo validation Macro-F1 improvement."
            )

            print(
                f"  Patience: "
                f"{rounds_without_improvement}/"
                f"{patience}"
            )

        # =============================================================
        # SAVE INTERMEDIATE LOG
        # =============================================================

        log_df = pd.DataFrame(
            log_rows
        )

        log_df.to_csv(
            log_path,
            index=False,
        )

        # =============================================================
        # EARLY STOPPING
        # =============================================================

        if (
            rounds_without_improvement
            >= patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"No improvement for "
                f"{patience} rounds."
            )

            break

        # =============================================================
        # GPU MEMORY
        # =============================================================

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / (1024 ** 2)
            )

            reserved = (
                torch.cuda.memory_reserved()
                / (1024 ** 2)
            )

            print(
                f"\nGPU memory:"
            )

            print(
                f"  Allocated: "
                f"{allocated:.2f} MB"
            )

            print(
                f"  Reserved : "
                f"{reserved:.2f} MB"
            )

    # =================================================================
    # SAVE FINAL GLOBAL MODEL
    # =================================================================

    print("\n" + "=" * 60)
    print(
        "SAVING FINAL GLOBAL MODEL"
    )
    print("=" * 60)

    save_checkpoint(
        path=final_model_path,
        model=global_model,
        round_number=round_number,
        val_loss=val_loss,
        val_acc=val_acc,
        val_macro_f1=val_macro_f1,
        best_val_f1=best_val_f1,
        config_snapshot={
            "architecture":
                "ConvNeXt-Tiny V2",
            "experiment":
                experiment_tag,
            "algorithm":
                "Sample-weighted FedAvg",
            "initialization":
                "ImageNet pretrained",
            "client_adaptive_learning_rate":
                True,
            "lr_adaptation":
                "automatic_exp_accuracy_delta",
            "lr_adaptation_rate":
                float(lr_adaptation_rate),
            "min_learning_rate":
                float(min_learning_rate),
            "max_learning_rate":
                float(max_learning_rate),
            "local_class_balanced_loss":
                True,
            "local_class_weighting":
                "smoothed_capped_inverse_frequency_per_client",
            "local_class_balanced_sampling":
                True,
            "sampling_method":
                "WeightedRandomSampler",
            "sampling_replacement":
                True,
            "sampler_weight_power":
                float(getattr(config, "FED_SAMPLER_WEIGHT_POWER", 0.5)),
            "sampler_min_weight":
                float(getattr(config, "FED_MIN_SAMPLER_WEIGHT", 0.25)),
            "sampler_max_weight":
                float(getattr(config, "FED_MAX_SAMPLER_WEIGHT", 5.0)),
            "accuracy_threshold":
                "disabled",
            "lr_update_rule":
                "next_lr = current_lr * exp(-adaptation_rate * accuracy_delta)",
        },
        client_learning_rates=client_learning_rates,
    )

    print(
        f"\nFinal global checkpoint saved:\n"
        f"  {final_model_path}"
    )

    # =================================================================
    # SAVE TRAINING LOG
    # =================================================================

    log_df = pd.DataFrame(
        log_rows
    )

    log_df.to_csv(
        log_path,
        index=False,
    )

    print(
        f"\nTraining log saved:\n"
        f"  {log_path}"
    )

    # =================================================================
    # TRAINING CURVES
    # =================================================================

    curve_path = os.path.join(
        plots_dir,
        f"federated_convnext_{experiment_tag}_training_curve.png",
    )

    plot_training_curves(
        log_df,
        curve_path,
    )

    # =================================================================
    # LOAD BEST GLOBAL CHECKPOINT
    # =================================================================

    print("\n" + "=" * 60)
    print(
        "LOADING BEST FEDERATED CHECKPOINT"
    )
    print("=" * 60)

    if not os.path.exists(
        best_model_path
    ):

        raise FileNotFoundError(
            "\nBest federated checkpoint was not created:\n"
            f"{best_model_path}"
        )

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    global_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    print(
        f"Best round: "
        f"{checkpoint['round']}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint['val_acc']:.4f}"
    )

    print(
        f"Best validation loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    # =================================================================
    # FINAL TEST EVALUATION
    # =================================================================

    print("\n" + "=" * 60)
    print(
        "FINAL FEDERATED TEST EVALUATION"
    )
    print("=" * 60)

    (
        test_loss,
        test_acc,
        test_macro_f1,
        test_preds,
        test_labels,
    ) = evaluate(
        model=global_model,
        loader=test_loader,
        device=device,
        criterion=eval_criterion,
    )

    print(
        f"\nTest Loss     : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_acc:.4f}"
    )

    print(
        f"Test Macro-F1 : "
        f"{test_macro_f1:.4f}"
    )

    # =================================================================
    # CLASSIFICATION REPORT
    # =================================================================

    print(
        "\nClassification Report:"
    )

    report = classification_report(
        test_labels,
        test_preds,
        labels=list(
            range(config.NUM_CLASSES)
        ),
        target_names=config.CLASS_LIST,
        zero_division=0,
    )

    print(report)

    report_path = os.path.join(
        log_dir,
        f"federated_convnext_{experiment_tag}_test_classification_report.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FEDERATED CONVNEXT-TINY V2 - 11A-v4 AUGMENTED + CLASS-BALANCED SAMPLING "
            "TEST CLASSIFICATION REPORT\n\n"
        )

        f.write(report)

        f.write(
            f"\nTest Loss: "
            f"{test_loss:.6f}\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{test_acc:.6f}\n"
        )

        f.write(
            f"Test Macro-F1: "
            f"{test_macro_f1:.6f}\n"
        )

        f.write(
            f"Best Federated Round: "
            f"{best_round}\n"
        )

        f.write(
            f"Best Validation Macro-F1: "
            f"{best_val_f1:.6f}\n"
        )

    print(
        f"\nSaved classification report:\n"
        f"  {report_path}"
    )

    # =================================================================
    # PER-CLASS F1
    # =================================================================

    per_class_f1 = calculate_per_class_f1(
        test_labels,
        test_preds,
    )

    print(
        "\nTest F1 by class:"
    )

    for cls, score in (
        per_class_f1.items()
    ):

        print(
            f"  {cls:6s}: "
            f"{score:.4f}"
        )

    per_class_df = pd.DataFrame(
        {
            "class": list(
                per_class_f1.keys()
            ),

            "f1_score": list(
                per_class_f1.values()
            ),
        }
    )

    per_class_path = os.path.join(
        log_dir,
        f"federated_convnext_{experiment_tag}_per_class_f1.csv",
    )

    per_class_df.to_csv(
        per_class_path,
        index=False,
    )

    print(
        f"\nSaved per-class F1:\n"
        f"  {per_class_path}"
    )

    # =================================================================
    # CONFUSION MATRIX
    # =================================================================

    confusion_path = os.path.join(
        plots_dir,
        f"federated_convnext_{experiment_tag}_confusion_matrix.png",
    )

    plot_confusion_matrix(
        test_labels,
        test_preds,
        confusion_path,
    )

    # =================================================================
    # CENTRALIZED COMPARISON
    # =================================================================

    print("\n" + "=" * 60)
    print(
        "CENTRALIZED vs FEDERATED CONVNEXT"
    )
    print("=" * 60)

    centralized_report_path = os.path.join(
        log_dir,
        "centralized_convnext_test_classification_report_v2.txt",
    )

    centralized_acc = None
    centralized_f1 = None

    if os.path.exists(
        centralized_report_path
    ):

        try:

            with open(
                centralized_report_path,
                "r",
                encoding="utf-8",
            ) as f:

                text = f.read()

            for line in text.splitlines():

                line = line.strip()

                if line.startswith(
                    "Test Accuracy:"
                ):

                    centralized_acc = float(
                        line.split(":")[-1]
                    )

                elif line.startswith(
                    "Test Macro-F1:"
                ):

                    centralized_f1 = float(
                        line.split(":")[-1]
                    )

        except Exception as exc:

            print(
                "\nCould not automatically read "
                "centralized ConvNeXt results:"
            )

            print(exc)

    print(
        "\nCentralized ConvNeXt-Tiny V2:"
    )

    if centralized_acc is not None:

        print(
            f"  Test Accuracy : "
            f"{centralized_acc:.4f}"
        )

    else:

        print(
            "  Test Accuracy : "
            "not automatically available"
        )

    if centralized_f1 is not None:

        print(
            f"  Test Macro-F1 : "
            f"{centralized_f1:.4f}"
        )

    else:

        print(
            "  Test Macro-F1 : "
            "not automatically available"
        )

    print(
        "\nFederated ConvNeXt-Tiny V2:"
    )

    print(
        f"  Test Accuracy : "
        f"{test_acc:.4f}"
    )

    print(
        f"  Test Macro-F1 : "
        f"{test_macro_f1:.4f}"
    )

    if centralized_acc is not None:

        accuracy_difference = (
            test_acc
            - centralized_acc
        )

        print(
            f"\nAccuracy difference "
            f"(FL - Centralized): "
            f"{accuracy_difference:+.4f}"
        )

    if centralized_f1 is not None:

        f1_difference = (
            test_macro_f1
            - centralized_f1
        )

        print(
            f"Macro-F1 difference "
            f"(FL - Centralized): "
            f"{f1_difference:+.4f}"
        )

    # =================================================================
    # FINAL GPU MEMORY
    # =================================================================

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 2)
        )

        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 2)
        )

        print("\nGPU memory:")

        print(
            f"  Allocated       : "
            f"{allocated:.2f} MB"
        )

        print(
            f"  Reserved        : "
            f"{reserved:.2f} MB"
        )

        print(
            f"  Peak allocated  : "
            f"{peak_allocated:.2f} MB"
        )

        print(
            f"  Peak reserved   : "
            f"{peak_reserved:.2f} MB"
        )

    # =================================================================
    # TOTAL TIME
    # =================================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print("\n" + "=" * 60)
    print(
        "PHASE 2 - STEP 11A-v4 COMPLETE ✓"
    )
    print("=" * 60)

    print(
        f"Total time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print("\n" + "-" * 60)
    print(
        "FEDERATED CONVNEXT-TINY V2 - 11A-v4 RESULTS"
    )
    print("-" * 60)

    print(
        f"Best Val Macro-F1 : "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Best Val Accuracy : "
        f"{best_val_acc:.4f}"
    )

    print(
        f"Best Round        : "
        f"{best_round}"
    )

    print(
        f"Test Accuracy     : "
        f"{test_acc:.4f}"
    )

    print(
        f"Test Macro-F1     : "
        f"{test_macro_f1:.4f}"
    )

    print("-" * 60)

    print("\nSaved outputs:")

    print(
        f"  Best global model:\n"
        f"    {best_model_path}"
    )

    print(
        f"  Final global model:\n"
        f"    {final_model_path}"
    )

    print(
        f"  Training log:\n"
        f"    {log_path}"
    )

    print(
        f"  Classification report:\n"
        f"    {report_path}"
    )

    print(
        f"  Per-class F1:\n"
        f"    {per_class_path}"
    )

    print(
        f"  Training curves:\n"
        f"    {curve_path}"
    )

    print(
        f"  Confusion matrix:\n"
        f"    {confusion_path}"
    )

    print("\n" + "=" * 60)
    print(
        "EXPERIMENT STATUS"
    )
    print("=" * 60)

    print(
        "\nCentralized ConvNeXt-Tiny V2:"
    )

    if centralized_acc is not None:
        print(
            f"  Accuracy : "
            f"{centralized_acc:.4f}"
        )
    else:
        print(
            "  Accuracy : see centralized results"
        )

    if centralized_f1 is not None:
        print(
            f"  Macro-F1 : "
            f"{centralized_f1:.4f}"
        )
    else:
        print(
            "  Macro-F1 : see centralized results"
        )

    print(
        "\nFederated ConvNeXt-Tiny V2:"
    )

    print(
        f"  Accuracy : "
        f"{test_acc:.4f}"
    )

    print(
        f"  Macro-F1 : "
        f"{test_macro_f1:.4f}"
    )

    print(
        "\nNext experimental stages:"
    )

    print(
        "  12A - Federated ConvNeXt + Differential Privacy"
    )

    print(
        "  13A - Federated ConvNeXt + DP-SGD"
    )

    print(
        "  14A - Privacy/utility comparison"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()