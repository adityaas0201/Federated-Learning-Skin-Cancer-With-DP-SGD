

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
# ADAPTIVE FEDPROX PROXIMAL COEFFICIENT
# =====================================================================

def compute_adaptive_fedprox_mu(client_loader, base_learning_rate, local_epochs):
    """Automatically calculate a client-specific FedProx coefficient."""
    if not hasattr(client_loader.dataset, "df"):
        raise AttributeError(
            "Client dataset must expose dataset.df for adaptive FedProx mu."
        )

    labels = client_loader.dataset.df["label"].tolist()
    if not labels:
        raise ValueError("Cannot calculate adaptive FedProx mu for an empty client.")

    counts = np.asarray(
        [sum(1 for label in labels if label == cls) for cls in config.CLASS_LIST],
        dtype=np.float64,
    )
    present = counts[counts > 0]
    probabilities = present / present.sum()

    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    max_entropy = float(np.log(len(config.CLASS_LIST)))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    heterogeneity = float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))

    # Automatic scale: no fixed FEDPROX_MU is required.  The optimizer
    # learning-rate scale and local epoch count determine the magnitude;
    # local class-distribution heterogeneity determines the strength.
    optimizer_scale = float(np.sqrt(max(base_learning_rate, 1e-12)))
    epoch_scale = float(1.0 + 0.5 * max(int(local_epochs) - 1, 0))
    mu = optimizer_scale * epoch_scale * heterogeneity

    # Conservative numerical bounds for stability on the existing pipeline.
    mu = float(np.clip(mu, 1e-4, 5e-2))
    return mu, heterogeneity


def compute_client_model_drift(local_state, global_state):
    """Compute normalized L2 parameter drift after local training."""
    squared_difference = 0.0
    squared_global = 0.0

    for name, global_tensor in global_state.items():
        if is_fedbn_state_key(name):
            continue
        if not torch.is_floating_point(global_tensor):
            continue
        local_tensor = local_state[name]
        g = global_tensor.float()
        w = local_tensor.float()
        squared_difference += float(torch.sum((w - g) ** 2).item())
        squared_global += float(torch.sum(g ** 2).item())

    return float(np.sqrt(squared_difference) / (np.sqrt(squared_global) + 1e-12))


# =====================================================================
# V12.1 FEDBN + CORRECTED DISCRIMINATIVE LEARNING RATE HELPERS
# =====================================================================

# FedBN state keys are discovered from the actual model at runtime.
# Only BatchNorm buffers/affine parameters are kept client-local; all
# ConvNeXt and non-BN parameters continue through the existing FedYogi path.
FEDBN_STATE_KEYS = set()

def discover_fedbn_state_keys(model):
    keys = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            prefix = f"{module_name}." if module_name else ""
            for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                key = prefix + suffix
                if key in model.state_dict():
                    keys.add(key)
    return keys

def is_fedbn_state_key(key):
    return key in FEDBN_STATE_KEYS

def apply_client_bn_state(model, client_bn_state):
    if client_bn_state:
        current = model.state_dict()
        for key, value in client_bn_state.items():
            if key in current:
                current[key].copy_(value.to(current[key].device, dtype=current[key].dtype))

def extract_client_bn_state(model):
    state = model.state_dict()
    return {key: state[key].detach().cpu().clone() for key in FEDBN_STATE_KEYS if key in state}

def average_bn_states(client_bn_states, weights, reference_state):
    """Create a server-side evaluation BN state without making BN global during training."""
    if not FEDBN_STATE_KEYS or not client_bn_states:
        return {}
    result = {}
    total_weight = float(sum(weights))
    if total_weight <= 0:
        total_weight = float(len(client_bn_states))
        weights = [1.0] * len(client_bn_states)
    for key in FEDBN_STATE_KEYS:
        if key not in reference_state:
            continue
        values = [state[key] for state in client_bn_states if key in state]
        if not values:
            continue
        # Integer num_batches_tracked cannot be meaningfully averaged as an integer.
        if not torch.is_floating_point(values[0]):
            result[key] = values[0].clone()
            continue
        acc = torch.zeros_like(values[0], dtype=torch.float32)
        for value, weight in zip(values, weights):
            acc.add_(value.float(), alpha=float(weight))
        result[key] = (acc / total_weight).to(dtype=reference_state[key].dtype)
    return result


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
    proximal_mu=None,
    min_learning_rate=1e-7,
    max_learning_rate=None,
    lr_adaptation_rate=1.0,
    backbone_lr_ratio=0.10,
    client_bn_state=None,
):
    """
    Train one client from the current global model.

    FedProx local objective:

        L = L_local + (mu / 2) * ||w - w_global||^2

    The global weights captured at the START of the round are kept fixed
    throughout this client's local training.

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
    if proximal_mu is None:
        raise ValueError("An adaptive FedProx mu must be supplied.")
    if proximal_mu < 0:
        raise ValueError("proximal_mu must be >= 0.")

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

    # FedBN: restore this client's private BN state after loading the shared
    # global model. BN parameters/statistics never become global training state.
    apply_client_bn_state(local_model, client_bn_state)

    local_model.train()

    # ---------------------------------------------------------------
    # FEDPROX GLOBAL REFERENCE
    # ---------------------------------------------------------------
    # Fixed copy of the round-start global parameters. These are never
    # updated during local training.
    global_parameter_reference = {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in global_state.items()
        if name in dict(local_model.named_parameters())
        and not is_fedbn_state_key(name)
    }

    if not global_parameter_reference:
        raise ValueError(
            "FedProx could not match global parameters to local model parameters."
        )

    # ---------------------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------------------

    # Discriminative LR: pretrained ConvNeXt backbone receives a smaller LR,
    # while the task-specific handcrafted/fusion/classification layers adapt faster.
    backbone_parameters = list(local_model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_parameters}
    head_parameters = [p for p in local_model.parameters() if id(p) not in backbone_ids]
    backbone_lr = float(learning_rate) * float(backbone_lr_ratio)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": float(learning_rate)},
        ],
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

        epoch_start_lr = optimizer.param_groups[1]["lr"]

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
                    classification_loss = criterion(logits, labels)

                    proximal_term = torch.zeros(
                        (), device=device, dtype=classification_loss.dtype
                    )
                    for name, parameter in local_model.named_parameters():
                        if parameter.requires_grad and name in global_parameter_reference:
                            proximal_term = proximal_term + torch.sum(
                                (parameter - global_parameter_reference[name]) ** 2
                            )

                    loss = classification_loss + (float(proximal_mu) / 2.0) * proximal_term

                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Client {client_id} produced a non-finite local loss."
                        )

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
                classification_loss = criterion(logits, labels)

                proximal_term = torch.zeros(
                    (), device=device, dtype=classification_loss.dtype
                )
                for name, parameter in local_model.named_parameters():
                    if parameter.requires_grad and name in global_parameter_reference:
                        proximal_term = proximal_term + torch.sum(
                            (parameter - global_parameter_reference[name]) ** 2
                        )

                loss = classification_loss + (float(proximal_mu) / 2.0) * proximal_term

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Client {client_id} produced a non-finite local loss."
                    )

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
                lr=f"{optimizer.param_groups[1]['lr']:.2e}",
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

        current_lr = optimizer.param_groups[1]["lr"]
        next_lr = current_lr * multiplier

        next_lr = max(
            min_learning_rate,
            min(max_learning_rate, next_lr),
        )

        optimizer.param_groups[1]["lr"] = next_lr
        optimizer.param_groups[0]["lr"] = next_lr * float(backbone_lr_ratio)

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

    final_learning_rate = optimizer.param_groups[1]["lr"]  # task-head LR; group 0 is the 0.10x backbone LR
    # V12.1 invariant: persisted client LR is always the task-head LR.
    # The backbone LR remains final_learning_rate * backbone_lr_ratio.

    local_state = {
        key: value.detach().cpu().clone()
        for key, value in local_model.state_dict().items()
    }
    local_bn_state = extract_client_bn_state(local_model)

    if len(local_bn_state) != len(FEDBN_STATE_KEYS):
        raise RuntimeError(
            f"Client {client_id}: incomplete FedBN state ({len(local_bn_state)}/{len(FEDBN_STATE_KEYS)} keys)."
        )

    del local_model
    del optimizer
    del scaler
    del global_parameter_reference

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        local_state,
        local_bn_state,
        final_sample_count,
        final_loss,
        final_accuracy,
        final_learning_rate,
        history,
    )


# =====================================================================
# v11 AUTOMATIC FEDYOGI SERVER + SMART CLIENT AGGREGATION
# =====================================================================

# v11 design:
#   * Server LR adapts from update magnitude + directional volatility.
#   * beta1 stays fixed at 0.90.
#   * beta2 adapts from directional volatility and is smoothed.
#   * Client aggregation uses sqrt(sample-size) + class entropy +
#     update reliability + observed client drift.
#   * The aggregation multiplier is bounded so large clients remain useful
#     and small/heterogeneous clients are not allowed to dominate.
#   * The underlying Dirichlet partition is NOT changed.

V11_FEDYOGI_INITIAL_LR = 1e-3
V11_FEDYOGI_MIN_LR = 1e-4
V11_FEDYOGI_MAX_LR = 3e-3
V11_FEDYOGI_BETA1 = 0.90
V11_FEDYOGI_INITIAL_BETA2 = 0.99
V11_FEDYOGI_MIN_BETA2 = 0.95
V11_FEDYOGI_MAX_BETA2 = 0.999
V11_FEDYOGI_EPSILON = 1e-8
V11_FEDYOGI_BETA2_SMOOTHING = 0.20

V11_CLIENT_WEIGHT_GAMMA = 0.50
V11_CLIENT_ENTROPY_STRENGTH = 0.50
V11_CLIENT_RELIABILITY_STRENGTH = 0.50
V11_CLIENT_DRIFT_STRENGTH = 0.50
V11_CLIENT_WEIGHT_MIN = 0.50
V11_CLIENT_WEIGHT_MAX = 2.00


# ---------------------------------------------------------------------
# Basic tensor statistics
# ---------------------------------------------------------------------
def _state_rms(values):
    rms_values = []
    for value in values:
        value = value.float()
        if not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite tensor encountered while computing RMS.")
        rms_values.append(float(torch.sqrt(torch.mean(value.square())).item()))
    if not rms_values:
        raise ValueError("No floating-point tensors available for RMS calculation.")
    return float(np.sqrt(np.mean(np.square(rms_values))))


def _flatten_state_delta(delta):
    parts = []
    for key, value in delta.items():
        if is_fedbn_state_key(key):
            continue
        if torch.is_floating_point(value):
            parts.append(value.float().reshape(-1))
    if not parts:
        raise ValueError("Cannot flatten an empty state delta.")
    return torch.cat(parts)


def _cosine_similarity(a, b):
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 1e-12:
        return 1.0
    value = torch.dot(a, b) / denom
    return float(torch.clamp(value, -1.0, 1.0).item())


def _client_delta(global_state, client_state):
    delta = {}
    for key, global_tensor in global_state.items():
        if is_fedbn_state_key(key):
            continue
        if not torch.is_floating_point(global_tensor):
            continue
        current = client_state[key].float()
        base = global_tensor.float()
        if not torch.isfinite(current).all():
            raise FloatingPointError(f"Non-finite client state detected in '{key}'.")
        d = current - base
        if not torch.isfinite(d).all():
            raise FloatingPointError(f"Non-finite client delta detected in '{key}'.")
        delta[key] = d
    return delta


def _client_class_entropy(client_loader):
    labels = client_loader.dataset.df["label"].tolist()
    counts = np.asarray(
        [sum(1 for label in labels if label == cls) for cls in config.CLASS_LIST],
        dtype=np.float64,
    )
    present = counts[counts > 0]
    if present.size <= 1:
        return 0.0
    probabilities = present / present.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return float(np.clip(entropy / np.log(config.NUM_CLASSES), 0.0, 1.0))


def compute_v11_client_weights(
    global_state,
    client_states,
    client_sample_counts,
    client_entropies,
    client_drifts,
):
    """
    Two-pass conservative client weighting.

    Base:
        N_i ** gamma

    Multipliers:
        entropy diversity: [0.75, 1.25]
        reliability:       bounded from cosine similarity to provisional mean
        drift:              bounded around the median client drift
        final multiplier:   [CLIENT_WEIGHT_MIN, CLIENT_WEIGHT_MAX]

    The result is normalized to sum to one.  Importantly, sampler draws are
    never used here; client_sample_counts are the true underlying dataset sizes.
    """
    n_clients = len(client_states)
    if not (
        len(client_sample_counts) == n_clients
        and len(client_entropies) == n_clients
        and len(client_drifts) == n_clients
    ):
        raise ValueError("Client metadata lengths do not match client states.")
    if n_clients == 0:
        raise ValueError("No clients supplied for aggregation.")

    gamma = float(V11_CLIENT_WEIGHT_GAMMA)
    entropy_strength = float(V11_CLIENT_ENTROPY_STRENGTH)
    reliability_strength = float(V11_CLIENT_RELIABILITY_STRENGTH)
    drift_strength = float(V11_CLIENT_DRIFT_STRENGTH)
    weight_min = float(V11_CLIENT_WEIGHT_MIN)
    weight_max = float(V11_CLIENT_WEIGHT_MAX)

    if gamma <= 0 or weight_min <= 0 or weight_max < weight_min:
        raise ValueError("Invalid v11 client-weight configuration.")

    base_weights = np.asarray(
        [max(float(n), 1.0) ** gamma for n in client_sample_counts],
        dtype=np.float64,
    )

    # First pass: size + diversity only.
    provisional = base_weights * np.asarray(
        [1.0 + entropy_strength * (float(h) - 0.5) for h in client_entropies],
        dtype=np.float64,
    )
    provisional = np.maximum(provisional, 1e-12)
    provisional /= provisional.sum()

    # Provisional aggregate direction for reliability scoring.
    client_vectors = [_flatten_state_delta(_client_delta(global_state, s)) for s in client_states]
    provisional_vector = sum(
        float(w) * v for w, v in zip(provisional, client_vectors)
    )

    similarities = np.asarray(
        [_cosine_similarity(v, provisional_vector) for v in client_vectors],
        dtype=np.float64,
    )
    # cosine -1..1 -> reliability .5..1.0; strength can blend it toward 1.
    raw_reliability = 0.75 + 0.25 * similarities
    reliability = 1.0 + reliability_strength * (raw_reliability - 1.0)

    finite_drifts = np.asarray([max(float(d), 1e-12) for d in client_drifts], dtype=np.float64)
    median_drift = float(np.median(finite_drifts))
    drift_ratio = finite_drifts / max(median_drift, 1e-12)
    # Low drift gets at most 1.25x; high drift is penalized conservatively.
    raw_drift = np.sqrt(1.0 / np.maximum(drift_ratio, 1e-12))
    raw_drift = np.clip(raw_drift, 0.75, 1.25)
    drift_factor = 1.0 + drift_strength * (raw_drift - 1.0)

    multipliers = np.asarray(
        [
            float(np.clip(
                1.0
                * (1.0 + entropy_strength * (float(h) - 0.5))
                * float(r)
                * float(d),
                weight_min,
                weight_max,
            ))
            for h, r, d in zip(client_entropies, reliability, drift_factor)
        ],
        dtype=np.float64,
    )

    final_weights = base_weights * multipliers
    final_weights = np.maximum(final_weights, 1e-12)
    final_weights /= final_weights.sum()

    diagnostics = []
    for idx in range(n_clients):
        diagnostics.append({
            "base_sqrt_size": float(base_weights[idx]),
            "entropy": float(client_entropies[idx]),
            "cosine_reliability": float(similarities[idx]),
            "drift": float(client_drifts[idx]),
            "final_multiplier": float(multipliers[idx]),
            "normalized_weight": float(final_weights[idx]),
        })

    return final_weights.tolist(), diagnostics


class AutomaticFedYogiV11:
    """FedYogi with automatic server LR, beta2, and smart client weighting."""

    def __init__(self):
        self.server_lr = float(V11_FEDYOGI_INITIAL_LR)
        self.min_lr = float(V11_FEDYOGI_MIN_LR)
        self.max_lr = float(V11_FEDYOGI_MAX_LR)
        self.beta1 = float(V11_FEDYOGI_BETA1)
        self.beta2 = float(V11_FEDYOGI_INITIAL_BETA2)
        self.min_beta2 = float(V11_FEDYOGI_MIN_BETA2)
        self.max_beta2 = float(V11_FEDYOGI_MAX_BETA2)
        self.epsilon = float(V11_FEDYOGI_EPSILON)
        self.beta2_smoothing = float(V11_FEDYOGI_BETA2_SMOOTHING)

        self.step_count = 0
        self.first_moment = {}
        self.second_moment = {}

        self.previous_delta_vector = None
        self.update_rms_ema = None
        self.volatility_ema = None

        self.last_update_rms = 0.0
        self.last_direction_rms = 0.0
        self.last_applied_update_rms = 0.0
        self.last_volatility = 0.0
        self.last_beta2_target = self.beta2
        self.last_client_weights = []
        self.last_client_weight_diagnostics = []

    @staticmethod
    def _rms(tensor):
        tensor = tensor.float()
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("RMS requested for a tensor containing NaN/Inf.")
        return float(torch.sqrt(torch.mean(tensor.square())).item())

    def _adapt_hyperparameters(self, current_update_rms, current_vector):
        # Directional volatility: 0 = same direction, 1 = orthogonal,
        # 2 = opposite direction.
        if self.previous_delta_vector is None:
            volatility = 0.0
        else:
            cosine = _cosine_similarity(current_vector, self.previous_delta_vector)
            volatility = float(np.clip(1.0 - cosine, 0.0, 2.0))

        if self.update_rms_ema is None:
            magnitude_ratio = 1.0
            self.update_rms_ema = float(current_update_rms)
        else:
            magnitude_ratio = float(
                current_update_rms / max(self.update_rms_ema, 1e-12)
            )
            # Slow EMA so the controller does not chase one noisy round.
            self.update_rms_ema = 0.80 * self.update_rms_ema + 0.20 * current_update_rms

        # Automatic LR:
        #   stable + small update  -> gently increase
        #   stable + large update  -> gently decrease
        #   volatility             -> decrease more
        # The log-ratio keeps the response symmetric and bounded.
        size_signal = float(np.clip(np.log(max(magnitude_ratio, 1e-6)), -3.0, 3.0))
        volatility_signal = float(np.clip(volatility, 0.0, 2.0))
        lr_factor = float(np.exp(-0.06 * size_signal - 0.12 * volatility_signal))
        lr_factor = float(np.clip(lr_factor, 0.85, 1.15))
        self.server_lr = float(np.clip(self.server_lr * lr_factor, self.min_lr, self.max_lr))

        # Stable direction -> beta2 high; volatile direction -> beta2 low.
        target_beta2 = self.max_beta2 - (
            self.max_beta2 - self.min_beta2
        ) * (volatility_signal / 2.0)
        target_beta2 = float(np.clip(target_beta2, self.min_beta2, self.max_beta2))
        self.beta2 = float(
            np.clip(
                (1.0 - self.beta2_smoothing) * self.beta2
                + self.beta2_smoothing * target_beta2,
                self.min_beta2,
                self.max_beta2,
            )
        )

        if self.volatility_ema is None:
            self.volatility_ema = volatility
        else:
            self.volatility_ema = 0.80 * self.volatility_ema + 0.20 * volatility

        self.last_volatility = volatility
        self.last_beta2_target = target_beta2
        self.previous_delta_vector = current_vector.detach().cpu().clone()

        return magnitude_ratio, volatility, lr_factor

    def step(
        self,
        global_state,
        client_states,
        client_sample_counts,
        client_entropies,
        client_drifts,
    ):
        if not client_states:
            raise ValueError("No client states supplied to AutomaticFedYogiV11.")

        weights, diagnostics = compute_v11_client_weights(
            global_state=global_state,
            client_states=client_states,
            client_sample_counts=client_sample_counts,
            client_entropies=client_entropies,
            client_drifts=client_drifts,
        )

        # Build the smart-weighted aggregate delta.
        delta = {}
        for key, global_tensor in global_state.items():
            # Strict FedBN: BN state is never aggregated by the server.
            if is_fedbn_state_key(key):
                continue
            if not torch.is_floating_point(global_tensor):
                continue
            base = global_tensor.float()
            result = torch.zeros_like(base, dtype=torch.float32)
            for state, weight in zip(client_states, weights):
                current = state[key].float()
                if not torch.isfinite(current).all():
                    raise FloatingPointError(f"Non-finite client state in '{key}'.")
                result.add_((current - base) * float(weight))
            if not torch.isfinite(result).all():
                raise FloatingPointError(f"Non-finite aggregate delta in '{key}'.")
            delta[key] = result

        current_update_rms = _state_rms(delta.values())
        current_vector = _flatten_state_delta(delta)
        magnitude_ratio, volatility, lr_factor = self._adapt_hyperparameters(
            current_update_rms, current_vector
        )

        candidate_step = self.step_count + 1
        bc1 = 1.0 - self.beta1 ** candidate_step
        bc2 = 1.0 - self.beta2 ** candidate_step
        if bc1 <= 0 or bc2 <= 0:
            raise FloatingPointError("Invalid FedYogi bias correction.")

        candidate_m = {}
        candidate_v = {}
        candidate_directions = {}
        direction_rms_values = []

        for key, current_delta in delta.items():
            if key in self.first_moment:
                old_m = self.first_moment[key].float()
                old_v = self.second_moment[key].float()
            else:
                old_m = torch.zeros_like(current_delta)
                old_v = torch.zeros_like(current_delta)

            new_m = self.beta1 * old_m + (1.0 - self.beta1) * current_delta
            delta_sq = current_delta.square()
            new_v = old_v - (1.0 - self.beta2) * torch.sign(old_v - delta_sq) * delta_sq
            new_v = torch.clamp(new_v, min=0.0)

            if not torch.isfinite(new_m).all() or not torch.isfinite(new_v).all():
                raise FloatingPointError(f"Non-finite FedYogi state in '{key}'.")

            m_hat = new_m / bc1
            v_hat = new_v / bc2
            direction = m_hat / (torch.sqrt(v_hat) + self.epsilon)
            if not torch.isfinite(direction).all():
                raise FloatingPointError(f"Non-finite FedYogi direction in '{key}'.")

            candidate_m[key] = new_m
            candidate_v[key] = new_v
            candidate_directions[key] = direction
            direction_rms_values.append(self._rms(direction))

        direction_rms = float(np.sqrt(np.mean(np.square(direction_rms_values))))
        if not np.isfinite(direction_rms) or direction_rms <= 0:
            raise FloatingPointError(f"Invalid FedYogi direction RMS: {direction_rms}")

        candidate_global_state = {}
        applied_rms_values = []
        for key, global_tensor in global_state.items():
            if key not in candidate_directions:
                candidate_global_state[key] = global_tensor.clone()
                continue
            candidate_tensor = global_tensor.float() + self.server_lr * candidate_directions[key]
            if not torch.isfinite(candidate_tensor).all():
                raise FloatingPointError(f"Server update produced NaN/Inf in '{key}'.")
            actual = candidate_tensor - global_tensor.float()
            applied_rms_values.append(self._rms(actual))
            candidate_global_state[key] = candidate_tensor.to(dtype=global_tensor.dtype)

        applied_rms = float(np.sqrt(np.mean(np.square(applied_rms_values))))
        if not np.isfinite(applied_rms):
            raise FloatingPointError("Invalid applied global-update RMS.")

        self.first_moment = {k: v.detach().cpu().clone() for k, v in candidate_m.items()}
        self.second_moment = {k: v.detach().cpu().clone() for k, v in candidate_v.items()}
        self.step_count = candidate_step
        self.last_update_rms = current_update_rms
        self.last_direction_rms = direction_rms
        self.last_applied_update_rms = applied_rms
        self.last_client_weights = [float(x) for x in weights]
        self.last_client_weight_diagnostics = diagnostics

        return candidate_global_state, sum(client_sample_counts), weights, {
            "magnitude_ratio": magnitude_ratio,
            "volatility": volatility,
            "lr_factor": lr_factor,
        }

    def state_dict(self):
        return {
            "step_count": int(self.step_count),
            "server_lr": float(self.server_lr),
            "min_lr": float(self.min_lr),
            "max_lr": float(self.max_lr),
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "min_beta2": float(self.min_beta2),
            "max_beta2": float(self.max_beta2),
            "epsilon": float(self.epsilon),
            "beta2_smoothing": float(self.beta2_smoothing),
            "first_moment": {k: v.detach().cpu().clone() for k, v in self.first_moment.items()},
            "second_moment": {k: v.detach().cpu().clone() for k, v in self.second_moment.items()},
            "previous_delta_vector": None if self.previous_delta_vector is None else self.previous_delta_vector.clone(),
            "update_rms_ema": self.update_rms_ema,
            "volatility_ema": self.volatility_ema,
            "last_update_rms": float(self.last_update_rms),
            "last_direction_rms": float(self.last_direction_rms),
            "last_applied_update_rms": float(self.last_applied_update_rms),
            "last_volatility": float(self.last_volatility),
            "last_beta2_target": float(self.last_beta2_target),
            "last_client_weights": list(self.last_client_weights),
            "last_client_weight_diagnostics": list(self.last_client_weight_diagnostics),
        }


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
    server_optimizer_state=None,
    evaluation_bn_state=None,
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
        # It is never averaged by the server.
        "client_learning_rates": {
            int(k): float(v)
            for k, v in (client_learning_rates or {}).items()
        },

        "server_optimizer_state": server_optimizer_state,
        "fedbn_evaluation_state": {
            key: value.detach().cpu().clone()
            for key, value in (evaluation_bn_state or {}).items()
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
        - keeps num_samples == len(dataset), so the server still uses the
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

    # One epoch draws 2x the original client sample count.
    # Replacement plus existing on-the-fly augmentation gives repeated
    # exposure to minority examples without creating files or changing the partition.
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=max(1, int(round(len(dataset) * 2.0))),
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
        f"  Sampler draws/epoch : {max(1, int(round(len(dataset) * 2.0)))} (2.0x original)"
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
    print("  ✓ Original client dataset count preserved for server aggregation")

    return balanced_loader


# =====================================================================
# MAIN
# =====================================================================

def main():
    start_time = time.perf_counter()

    print("=" * 72)
    print("PHASE 3 - STEP 11A-v12: FEDPROX + AUTOMATIC FEDYOGI + SMART AGGREGATION + FEDBN")
    print("CONVNEXT-TINY V2 FEDERATED TRAINING")
    print("=" * 72)

    device = get_device()
    seed = getattr(config, "RANDOM_SEED", 42)
    set_random_seeds(seed)

    # ---------------------------------------------------------------
    # v12 local recipe: V11 unchanged + discriminative LR + FedBN
    # ---------------------------------------------------------------
    local_epochs = 5
    local_sample_multiplier = 2.0
    backbone_lr_ratio = 0.10  # backbone = 10% of task-head LR
    num_rounds = int(getattr(config, "FED_NUM_ROUNDS", getattr(config, "NUM_FEDERATED_ROUNDS", 30)))
    clients_per_round = getattr(config, "FED_CLIENTS_PER_ROUND", getattr(config, "CLIENTS_PER_ROUND", None))
    if clients_per_round is None:
        clients_per_round = getattr(config, "NUM_CLIENTS", getattr(config, "NUM_FEDERATED_CLIENTS", 10))
    clients_per_round = int(clients_per_round)

    # V12.1 starts the task-specific/head LR at 1e-4 and persists the HEAD LR between rounds.
    # The pretrained ConvNeXt backbone receives 1e-5 via backbone_lr_ratio. The persisted LR is the HEAD LR, not the backbone LR.
    learning_rate = 1e-4
    weight_decay = float(getattr(config, "FED_WEIGHT_DECAY", getattr(config, "WEIGHT_DECAY", 1e-4)))
    min_learning_rate = 1e-5  # V12.1: keep task-head LR out of the 1e-6 range
    max_learning_rate = float(getattr(config, "FED_MAX_LEARNING_RATE", 2e-4))
    lr_adaptation_rate = 0.25  # V12.1: slower adaptation prevents premature LR collapse
    patience = int(getattr(config, "FED_EARLY_STOPPING_PATIENCE", getattr(config, "EARLY_STOPPING_PATIENCE", 8)))
    min_delta = float(getattr(config, "FED_MIN_DELTA", 1e-4))
    use_amp = bool(getattr(config, "USE_AMP", True)) and torch.cuda.is_available()
    grad_clip = 1.0

    if local_sample_multiplier != 2.0:
        raise ValueError("v12 requires local_sample_multiplier=2.0.")

    print("\n" + "=" * 72)
    print("V12.1 EXPERIMENT CONFIGURATION")
    print("=" * 72)
    print("Architecture              : ConvNeXt-Tiny V2")
    print(f"Federated rounds          : {num_rounds}")
    print(f"Local epochs              : {local_epochs}")
    print(f"Clients per round         : {clients_per_round}")
    print(f"Task-head base LR         : {learning_rate:.2e}")
    print(f"ConvNeXt backbone LR      : {learning_rate * backbone_lr_ratio:.2e} ({backbone_lr_ratio:.2f}x head LR)")
    print(f"Adaptive head LR bounds   : [{min_learning_rate:.2e}, {max_learning_rate:.2e}]")
    print("FedBN                     : enabled; client-local BN state")
    print("FedProx mu                : automatic per-client (existing method)")
    print(f"Local sampler             : WeightedRandomSampler, {local_sample_multiplier:.1f}x draws")
    print("Underlying Dirichlet      : unchanged natural non-IID partition")
    print("Augmentation              : existing on-the-fly augmentation, repeated by 2x sampling")
    print("Server optimizer          : Automatic FedYogi")
    print(f"Server LR                 : {V11_FEDYOGI_INITIAL_LR:.2e}, bounds [{V11_FEDYOGI_MIN_LR:.2e}, {V11_FEDYOGI_MAX_LR:.2e}]")
    print(f"FedYogi beta1             : {V11_FEDYOGI_BETA1:.3f} (fixed)")
    print(f"FedYogi beta2             : {V11_FEDYOGI_INITIAL_BETA2:.3f}, bounds [{V11_FEDYOGI_MIN_BETA2:.3f}, {V11_FEDYOGI_MAX_BETA2:.3f}]")
    print(f"Beta2 smoothing           : {V11_FEDYOGI_BETA2_SMOOTHING:.2f}")
    print("Client aggregation        : sqrt(N) + entropy + reliability + drift")
    print(f"Client weight multiplier  : [{V11_CLIENT_WEIGHT_MIN:.2f}, {V11_CLIENT_WEIGHT_MAX:.2f}]")
    print("Test set                  : final evaluation only")
    print("=" * 72)

    client_ids = get_client_ids()
    print(f"\nTotal clients configured: {len(client_ids)}")
    print(f"Client IDs: {client_ids}")

    model_dir = os.path.join(config.OUTPUT_DIR, "models")
    log_dir = os.path.join(config.OUTPUT_DIR, "training_logs")
    plots_dir = os.path.join(config.OUTPUT_DIR, "plots")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    experiment_tag = "11A_v12_1_corrected_lr_fedprox_autofedyogi_smartagg_2xaug_discriminative_fedbn"
    best_model_path = os.path.join(model_dir, f"federated_convnext_{experiment_tag}_global_best.pt")
    final_model_path = os.path.join(model_dir, f"federated_convnext_{experiment_tag}_global_final.pt")
    log_path = os.path.join(log_dir, f"federated_convnext_{experiment_tag}_training_log.csv")

    global_model = build_global_model(device)
    eval_criterion = build_criterion(device, class_weights=None)
    val_loader = build_eval_loader("val")
    test_loader = build_eval_loader("test")

    global_state = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    global FEDBN_STATE_KEYS
    FEDBN_STATE_KEYS = discover_fedbn_state_keys(global_model)
    if not FEDBN_STATE_KEYS:
        raise RuntimeError("V12 requires useful BatchNorm layers, but none were found.")
    print("\nFedBN state keys:")
    for key in sorted(FEDBN_STATE_KEYS):
        print(f"  {key}")

    initial_bn_state = {key: global_state[key].clone() for key in FEDBN_STATE_KEYS}
    client_bn_states = {int(cid): {key: value.clone() for key, value in initial_bn_state.items()} for cid in client_ids}

    server_optimizer = AutomaticFedYogiV11()
    print("\nAutomatic FedYogi initialized:")
    print(f"  LR    : {server_optimizer.server_lr:.6e}")
    print(f"  beta1 : {server_optimizer.beta1:.6f}")
    print(f"  beta2 : {server_optimizer.beta2:.6f}")

    client_learning_rates = {int(cid): learning_rate for cid in client_ids}

    # Existing adaptive FedProx state is retained exactly.
    adaptive_mu_state = {}
    adaptive_mu_heterogeneity = {}
    adaptive_mu_drift = {}

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_round = 0
    rounds_without_improvement = 0
    log_rows = []
    round_number = 0
    val_loss = float("inf")
    val_acc = 0.0
    val_macro_f1 = 0.0

    for round_number in range(1, num_rounds + 1):
        round_start = time.perf_counter()
        print("\n" + "=" * 72)
        print(f"FEDERATED ROUND {round_number}/{num_rounds}")
        print("=" * 72)

        selected_clients = select_clients(client_ids, round_number, clients_per_round)
        print(f"Selected clients: {selected_clients}")

        client_states = []
        client_sample_counts = []
        client_entropies = []
        client_drifts = []
        client_losses = []
        client_accuracies = []
        client_histories = {}
        participating_client_ids = []

        for client_id in selected_clients:
            print("\n" + "-" * 64)
            print(f"CLIENT {client_id}")
            print("-" * 64)

            client_loader = build_client_loader(client_id)
            original_sample_count = len(client_loader.dataset)
            if original_sample_count <= 0:
                print(f"WARNING: Client {client_id} contains zero samples. Skipping.")
                del client_loader
                continue

            actual_draws = max(1, int(round(original_sample_count * local_sample_multiplier)))
            print(f"Original client samples : {original_sample_count}")
            print(f"Sampler draws per epoch : {actual_draws} ({local_sample_multiplier:.1f}x)")

            client_class_weights = compute_client_class_weights(client_loader, device)
            local_criterion = build_criterion(device, class_weights=client_class_weights)

            adaptive_mu, heterogeneity = compute_adaptive_fedprox_mu(
                client_loader=client_loader,
                base_learning_rate=float(client_learning_rates[client_id]),
                local_epochs=local_epochs,
            )

            if client_id in adaptive_mu_state and client_id in adaptive_mu_drift:
                previous_mu = float(adaptive_mu_state[client_id])
                previous_drift = float(adaptive_mu_drift[client_id])
                reference_drift = float(np.mean(list(adaptive_mu_drift.values())))
                if reference_drift > 1e-12:
                    drift_ratio = previous_drift / reference_drift
                    adaptive_mu *= float(np.clip(drift_ratio, 0.5, 2.0))
                    adaptive_mu = float(np.clip(adaptive_mu, 1e-4, 5e-2))
                    adaptive_mu = float(np.sqrt(max(previous_mu, 1e-12) * adaptive_mu))

            adaptive_mu_state[client_id] = float(adaptive_mu)
            adaptive_mu_heterogeneity[client_id] = float(heterogeneity)

            print(f"Adaptive FedProx mu       : {adaptive_mu:.6e} (heterogeneity={heterogeneity:.4f})")
            print(f"Client LR entering round : {client_learning_rates[client_id]:.2e}")

            try:
                (
                    local_state,
                    local_bn_state,
                    _sampled_epoch_count,
                    client_loss,
                    client_acc,
                    final_client_lr,
                    client_history,
                ) = train_client(
                    global_state=global_state,
                    client_id=client_id,
                    client_loader=client_loader,
                    client_bn_state=client_bn_states.get(client_id),
                    device=device,
                    criterion=local_criterion,
                    use_amp=use_amp,
                    local_epochs=local_epochs,
                    learning_rate=float(client_learning_rates[client_id]),
                    weight_decay=weight_decay,
                    proximal_mu=float(adaptive_mu),
                    min_learning_rate=min_learning_rate,
                    max_learning_rate=max_learning_rate,
                    lr_adaptation_rate=lr_adaptation_rate,
                    backbone_lr_ratio=backbone_lr_ratio,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print("\nCUDA OUT OF MEMORY on Client", client_id)
                    print("Reduce the client batch size in config.py from 16 to 8 and rerun.")
                raise

            validate_state_dict(local_state, global_state)
            client_bn_states[client_id] = {key: value.clone() for key, value in local_bn_state.items()}
            drift = compute_client_model_drift(local_state, global_state)
            adaptive_mu_drift[client_id] = drift
            entropy = _client_class_entropy(client_loader)

            client_states.append(local_state)
            client_sample_counts.append(original_sample_count)  # TRUE dataset size, never 2x draws.
            client_entropies.append(entropy)
            client_drifts.append(drift)
            client_losses.append(client_loss)
            client_accuracies.append(client_acc)
            client_histories[client_id] = client_history
            participating_client_ids.append(client_id)
            client_learning_rates[client_id] = float(final_client_lr)

            print(f"\nClient {client_id} complete:")
            print(f"  True samples          : {original_sample_count}")
            print(f"  Sampler draws/epoch   : {actual_draws}")
            print(f"  Class entropy         : {entropy:.4f}")
            print(f"  Model drift           : {drift:.6e}")
            print(f"  Loss                  : {client_loss:.4f}")
            print(f"  Accuracy              : {client_acc:.4f}")
            print(f"  Final LR              : {final_client_lr:.2e}")

            del client_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not client_states:
            raise RuntimeError(f"No valid client updates were produced in round {round_number}.")

        # ---------------------------------------------------------------
        # Automatic FedYogi server update
        # ---------------------------------------------------------------
        print("\n" + "-" * 64)
        print("SERVER: AUTOMATIC FEDYOGI + SMART AGGREGATION + FEDBN")
        print("-" * 64)

        (
            new_global_state,
            total_client_samples,
            smart_weights,
            server_diagnostics,
        ) = server_optimizer.step(
            global_state=global_state,
            client_states=client_states,
            client_sample_counts=client_sample_counts,
            client_entropies=client_entropies,
            client_drifts=client_drifts,
        )

        print(f"\nTotal TRUE participating samples: {total_client_samples}")
        print("Smart aggregation weights:")
        for cid, count, weight, diag in zip(
            participating_client_ids,
            client_sample_counts,
            smart_weights,
            server_optimizer.last_client_weight_diagnostics,
        ):
            print(
                f"  Client {cid}: samples={count}, weight={weight:.6f}, "
                f"entropy={diag['entropy']:.4f}, cosine={diag['cosine_reliability']:.4f}, "
                f"drift={diag['drift']:.3e}, multiplier={diag['final_multiplier']:.4f}"
            )

        print(f"\nAutomatic server LR       : {server_optimizer.server_lr:.6e}")
        print(f"Server LR factor          : {server_diagnostics['lr_factor']:.6f}")
        print(f"Update magnitude ratio    : {server_diagnostics['magnitude_ratio']:.6f}")
        print(f"Update volatility         : {server_diagnostics['volatility']:.6f}")
        print(f"Automatic beta2           : {server_optimizer.beta2:.6f}")
        print(f"Beta2 target              : {server_optimizer.last_beta2_target:.6f}")
        print(f"Fixed beta1               : {server_optimizer.beta1:.6f}")
        print(f"Client-update RMS         : {server_optimizer.last_update_rms:.6e}")
        print(f"FedYogi direction RMS     : {server_optimizer.last_direction_rms:.6e}")
        print(f"Applied update RMS        : {server_optimizer.last_applied_update_rms:.6e}")

        validate_state_dict(new_global_state, global_state)
        global_state = new_global_state

        # FedBN training state stays client-local. For centralized validation
        # and final held-out testing, build a deterministic server-side BN
        # evaluation state from the participating clients' private BN states.
        participating_bn_states = [client_bn_states[cid] for cid in participating_client_ids]
        evaluation_bn_state = average_bn_states(
            participating_bn_states, smart_weights, global_state
        )
        eval_state = {key: value.clone() for key, value in global_state.items()}
        eval_state.update(evaluation_bn_state)
        global_model.load_state_dict(eval_state, strict=True)

        # ---------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------
        val_loss, val_acc, val_macro_f1, _, _ = evaluate(
            global_model, val_loader, device, eval_criterion
        )

        mean_client_loss = float(np.mean(client_losses))
        mean_client_acc = float(np.mean(client_accuracies))
        mean_client_lr_end = float(np.mean([client_learning_rates[c] for c in participating_client_ids]))
        round_time = time.perf_counter() - round_start

        print("\n" + "=" * 64)
        print(f"ROUND {round_number} RESULTS")
        print("=" * 64)
        print(f"  Participating clients : {len(client_states)}")
        print(f"  True client samples   : {total_client_samples}")
        print(f"  Mean client loss      : {mean_client_loss:.4f}")
        print(f"  Mean client accuracy  : {mean_client_acc:.4f}")
        print(f"  Validation loss       : {val_loss:.4f}")
        print(f"  Validation accuracy   : {val_acc:.4f}")
        print(f"  Validation Macro-F1   : {val_macro_f1:.4f}")
        print(f"  Mean client LR        : {mean_client_lr_end:.2e}")
        print(f"  Server LR             : {server_optimizer.server_lr:.2e}")
        print(f"  Beta2                 : {server_optimizer.beta2:.6f}")
        print(f"  Round time            : {round_time:.1f}s")

        log_rows.append({
            "round": round_number,
            "participating_clients": len(client_states),
            "participating_client_ids": str(participating_client_ids),
            "client_samples_true": total_client_samples,
            "sampler_multiplier": local_sample_multiplier,
            "mean_client_loss": mean_client_loss,
            "mean_client_acc": mean_client_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "mean_client_learning_rate_end": mean_client_lr_end,
            "client_learning_rates": str({int(k): float(v) for k, v in client_learning_rates.items()}),
            "client_entropies": str({int(c): float(e) for c, e in zip(participating_client_ids, client_entropies)}),
            "client_drifts": str({int(c): float(d) for c, d in zip(participating_client_ids, client_drifts)}),
            "smart_aggregation_weights": str({int(c): float(w) for c, w in zip(participating_client_ids, smart_weights)}),
            "smart_aggregation_diagnostics": str(server_optimizer.last_client_weight_diagnostics),
            "adaptive_fedprox_mu": str({int(k): float(v) for k, v in adaptive_mu_state.items()}),
            "adaptive_fedprox_heterogeneity": str({int(k): float(v) for k, v in adaptive_mu_heterogeneity.items()}),
            "adaptive_fedprox_drift": str({int(k): float(v) for k, v in adaptive_mu_drift.items()}),
            "fedyogi_server_learning_rate": float(server_optimizer.server_lr),
            "fedyogi_beta1": float(server_optimizer.beta1),
            "fedyogi_beta2": float(server_optimizer.beta2),
            "fedyogi_beta2_target": float(server_optimizer.last_beta2_target),
            "fedyogi_update_volatility": float(server_optimizer.last_volatility),
            "fedyogi_update_magnitude_ratio": float(server_diagnostics["magnitude_ratio"]),
            "fedyogi_lr_factor": float(server_diagnostics["lr_factor"]),
            "fedyogi_client_update_rms": float(server_optimizer.last_update_rms),
            "fedyogi_direction_rms": float(server_optimizer.last_direction_rms),
            "fedyogi_applied_update_rms": float(server_optimizer.last_applied_update_rms),
            "fedyogi_step": int(server_optimizer.step_count),
            "client_epoch_histories": str(client_histories),
            "round_time_sec": round_time,
        })

        # ---------------------------------------------------------------
        # Best checkpoint by validation Macro-F1
        # ---------------------------------------------------------------
        improvement = val_macro_f1 > best_val_f1 + min_delta
        if improvement:
            best_val_f1 = val_macro_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_round = round_number
            rounds_without_improvement = 0

            config_snapshot = {
                "architecture": "ConvNeXt-Tiny V2",
                "experiment": experiment_tag,
                "initialization": "ImageNet pretrained",
                "local_epochs": local_epochs,
                "local_sample_multiplier": local_sample_multiplier,
                "natural_non_iid_partition": "unchanged Dirichlet alpha=0.3 partition",
                "on_the_fly_augmentation": True,
                "sampling_method": "WeightedRandomSampler",
                "sampling_replacement": True,
                "sampler_draws": "2x original client dataset size per local epoch",
                "server_optimizer": "Automatic FedYogi v11 (retained in V12.1)",
                "server_lr_initial": V11_FEDYOGI_INITIAL_LR,
                "server_lr_bounds": [V11_FEDYOGI_MIN_LR, V11_FEDYOGI_MAX_LR],
                "server_lr_controller": "update magnitude ratio + directional volatility",
                "beta1": V11_FEDYOGI_BETA1,
                "beta2_initial": V11_FEDYOGI_INITIAL_BETA2,
                "beta2_bounds": [V11_FEDYOGI_MIN_BETA2, V11_FEDYOGI_MAX_BETA2],
                "beta2_controller": "directional volatility with exponential smoothing",
                "beta2_smoothing": V11_FEDYOGI_BETA2_SMOOTHING,
                "client_aggregation": "sqrt(sample_size) + entropy + reliability + drift",
                "client_weight_multiplier_bounds": [V11_CLIENT_WEIGHT_MIN, V11_CLIENT_WEIGHT_MAX],
                "client_weight_gamma": V11_CLIENT_WEIGHT_GAMMA,
                "client_entropy_strength": V11_CLIENT_ENTROPY_STRENGTH,
                "client_reliability_strength": V11_CLIENT_RELIABILITY_STRENGTH,
                "client_drift_strength": V11_CLIENT_DRIFT_STRENGTH,
                "fedprox_mu": "adaptive_per_client_existing_method",
                "client_adaptive_learning_rate": True,
                "discriminative_learning_rate": True,
                "task_head_initial_lr": learning_rate,
                "backbone_initial_lr": learning_rate * backbone_lr_ratio,
                "backbone_lr_ratio": backbone_lr_ratio,
                "fedbn": True,
                "fedbn_policy": "BN state private per client; excluded from server optimization and aggregation",
                "fedbn_evaluation": "smart-weighted average of participating client BN states for global validation/test",
                "lr_adaptation": "automatic_exp_accuracy_delta",
                "lr_adaptation_rate": lr_adaptation_rate,
                "min_learning_rate": min_learning_rate,
                "max_learning_rate": max_learning_rate,
                "weight_decay": weight_decay,
                "gradient_clip": grad_clip,
                "use_amp": use_amp,
                "test_usage": "final evaluation only",
            }

            save_checkpoint(
                best_model_path,
                global_model,
                round_number,
                val_loss,
                val_acc,
                val_macro_f1,
                best_val_f1,
                config_snapshot,
                client_learning_rates,
                server_optimizer.state_dict(),
                evaluation_bn_state,
            )
            print(f"\n✓ NEW BEST GLOBAL MODEL — round {best_round}, Val Macro-F1={best_val_f1:.4f}")
        else:
            rounds_without_improvement += 1
            print(f"\nNo validation Macro-F1 improvement. Patience {rounds_without_improvement}/{patience}")

        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        if rounds_without_improvement >= patience:
            print("\nEarly stopping triggered.")
            break

        if torch.cuda.is_available():
            print(
                f"GPU memory: allocated={torch.cuda.memory_allocated()/(1024**2):.2f} MB, "
                f"reserved={torch.cuda.memory_reserved()/(1024**2):.2f} MB"
            )

    # ---------------------------------------------------------------
    # Final checkpoint
    # ---------------------------------------------------------------
    save_checkpoint(
        final_model_path,
        global_model,
        round_number,
        val_loss,
        val_acc,
        val_macro_f1,
        best_val_f1,
        {
            "architecture": "ConvNeXt-Tiny V2",
            "experiment": experiment_tag,
            "server_optimizer": "Automatic FedYogi v11 (retained in V12.1)",
            "server_lr_bounds": [V11_FEDYOGI_MIN_LR, V11_FEDYOGI_MAX_LR],
            "beta2_bounds": [V11_FEDYOGI_MIN_BETA2, V11_FEDYOGI_MAX_BETA2],
            "local_sample_multiplier": local_sample_multiplier,
            "natural_non_iid_partition": "unchanged Dirichlet alpha=0.3 partition",
            "client_aggregation": "sqrt(sample_size) + entropy + reliability + drift",
            "fedprox_mu": "adaptive_per_client_existing_method",
            "discriminative_learning_rate": True,
            "task_head_initial_lr": learning_rate,
            "backbone_initial_lr": learning_rate * backbone_lr_ratio,
            "backbone_lr_ratio": backbone_lr_ratio,
            "fedbn": True,
            "fedbn_policy": "BN state private per client; excluded from server optimization and aggregation",
            "fedbn_evaluation": "smart-weighted average of participating client BN states for global validation/test",
        },
        client_learning_rates,
        server_optimizer.state_dict(),
        evaluation_bn_state,
    )
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    curve_path = os.path.join(plots_dir, f"federated_convnext_{experiment_tag}_training_curve.png")
    plot_training_curves(pd.DataFrame(log_rows), curve_path)

    # ---------------------------------------------------------------
    # Load best model and evaluate on held-out test set once.
    # ---------------------------------------------------------------
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best federated checkpoint was not created:\n{best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    global_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    saved_eval_bn_state = checkpoint.get("fedbn_evaluation_state", {})
    if saved_eval_bn_state:
        eval_state = global_model.state_dict()
        for key, value in saved_eval_bn_state.items():
            if key in eval_state:
                eval_state[key].copy_(value.to(device, dtype=eval_state[key].dtype))

    test_loss, test_acc, test_macro_f1, test_preds, test_labels = evaluate(
        global_model, test_loader, device, eval_criterion
    )

    report = classification_report(
        test_labels,
        test_preds,
        labels=list(range(config.NUM_CLASSES)),
        target_names=config.CLASS_LIST,
        zero_division=0,
    )
    print("\n" + "=" * 72)
    print("FINAL V12.1 TEST RESULTS")
    print("=" * 72)
    print(f"Best round        : {best_round}")
    print(f"Best Val Macro-F1 : {best_val_f1:.4f}")
    print(f"Best Val Accuracy : {best_val_acc:.4f}")
    print(f"Test Loss         : {test_loss:.4f}")
    print(f"Test Accuracy     : {test_acc:.4f}")
    print(f"Test Macro-F1     : {test_macro_f1:.4f}")
    print("\nClassification Report:\n")
    print(report)

    report_path = os.path.join(log_dir, f"federated_convnext_{experiment_tag}_test_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("FEDERATED CONVNEXT-TINY V2 - v12 AUTOMATIC FEDYOGI + DISCRIMINATIVE LR + FEDBN TEST REPORT\n\n")
        f.write(report)
        f.write(f"\nTest Loss: {test_loss:.6f}\n")
        f.write(f"Test Accuracy: {test_acc:.6f}\n")
        f.write(f"Test Macro-F1: {test_macro_f1:.6f}\n")
        f.write(f"Best Federated Round: {best_round}\n")
        f.write(f"Best Validation Macro-F1: {best_val_f1:.6f}\n")

    per_class_f1 = calculate_per_class_f1(test_labels, test_preds)
    per_class_df = pd.DataFrame({"class": list(per_class_f1.keys()), "f1_score": list(per_class_f1.values())})
    per_class_path = os.path.join(log_dir, f"federated_convnext_{experiment_tag}_per_class_f1.csv")
    per_class_df.to_csv(per_class_path, index=False)

    confusion_path = os.path.join(plots_dir, f"federated_convnext_{experiment_tag}_confusion_matrix.png")
    plot_confusion_matrix(test_labels, test_preds, confusion_path)

    # Centralized reference, if the existing report is available.
    centralized_report_path = os.path.join(log_dir, "centralized_convnext_test_classification_report_v2.txt")
    centralized_acc = None
    centralized_f1 = None
    if os.path.exists(centralized_report_path):
        try:
            text = Path(centralized_report_path).read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.strip().startswith("Test Accuracy:"):
                    centralized_acc = float(line.split(":")[-1])
                elif line.strip().startswith("Test Macro-F1:"):
                    centralized_f1 = float(line.split(":")[-1])
        except Exception:
            pass

    print("\n" + "=" * 72)
    print("V11 vs CENTRALIZED REFERENCE")
    print("=" * 72)
    print(f"V11 Test Accuracy : {test_acc:.4f}")
    print(f"V11 Test Macro-F1 : {test_macro_f1:.4f}")
    if centralized_acc is not None:
        print(f"Centralized Acc    : {centralized_acc:.4f}")
        print(f"Accuracy diff      : {test_acc - centralized_acc:+.4f}")
    if centralized_f1 is not None:
        print(f"Centralized Macro-F1: {centralized_f1:.4f}")
        print(f"Macro-F1 diff       : {test_macro_f1 - centralized_f1:+.4f}")

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 72)
    print("PHASE 3 - STEP 11A-v11 COMPLETE ✓")
    print("=" * 72)
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Best model : {best_model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Log        : {log_path}")
    print(f"Report     : {report_path}")
    print(f"Per-class F1: {per_class_path}")
    print(f"Curve      : {curve_path}")
    print(f"Confusion  : {confusion_path}")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()

