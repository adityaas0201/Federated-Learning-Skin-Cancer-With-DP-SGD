"""
09A_federated_client_mobilenet_fedprox.py
==========================================

PHASE 2 - STEP 9
FEDERATED CLIENT LOCAL TRAINING

Runs after:
    06_dataset.py
    07_model.py
    08_train_centralized.py

Purpose:
    Implements the local training logic for one simulated hospital/client
    during one federated learning round using FedProx.

Federated workflow:

    Global Model
         |
         | broadcast global weights
         v
    +---------+---------+---------+---------+---------+---------+
    | Client0 | Client1 | Client2 | Client3 | Client4 | Client5 |
    +---------+---------+---------+---------+---------+---------+
         |         |         |         |         |         |
       local     local     local     local     local     local
      training  training  training  training  training  training
         |         |         |         |         |         |
         +---------+---------+---------+---------+---------+
                              |
                              v
                    Updated local weights
                              |
                              v
                       Step 10: FedAvg


IMPORTANT:
    - No Differential Privacy in this step.
    - Uses FedProx local proximal regularization.
    - No noise is added.
    - No Differential Privacy is used in this file.

The purpose of this step is to establish clean and correct local
training before aggregation.

Design decisions:
    1. Every client starts from the SAME global model.
    2. Each client trains ONLY on its own local partition.
    3. A fresh optimizer is created for every client every round.
    4. Global class weights are shared across all clients.
    5. AMP and gradient clipping match the centralized baseline.
    6. Returned model weights are moved to CPU to save GPU memory.
    7. Per-local-epoch metrics are retained for research logging.
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
import torch
from tqdm import tqdm


# =====================================================================
# CONFIG PATH
# =====================================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs"
    )
)

import config


# =====================================================================
# FEDPROX CONFIGURATION
# =====================================================================

FEDPROX_MU = float(getattr(config, "FEDPROX_MU", 0.01))
GRADIENT_CLIP = float(getattr(config, "FEDERATED_GRADIENT_CLIP", 1.0))

if FEDPROX_MU < 0.0:
    raise ValueError("FEDPROX_MU must be >= 0.")
if GRADIENT_CLIP <= 0.0:
    raise ValueError("FEDERATED_GRADIENT_CLIP must be > 0.")


# =====================================================================
# LOAD SIBLING MODULES
# =====================================================================

def _load_module(filename, module_name):
    """
    Dynamically load a Python module located in the same src directory.
    """

    path = os.path.join(
        os.path.dirname(__file__),
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required module not found:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module:\n{path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


dataset_module = _load_module(
    "06_dataset.py",
    "dataset_module"
)

model_module = _load_module(
    "07_model.py",
    "model_module"
)

centralized_module = _load_module(
    "08_train_centralized.py",
    "centralized_module"
)


# =====================================================================
# REUSE EXISTING FUNCTIONS
# =====================================================================

build_client_loader = dataset_module.build_client_loader
HybridClassifier = model_module.HybridClassifier

# Reuse exactly the same training logic as Step 8.
compute_class_weights = centralized_module.compute_class_weights
train_one_epoch = centralized_module.train_one_epoch
evaluate = centralized_module.evaluate


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Note:
        Federated training can still have small numerical differences
        depending on CUDA/cuDNN behavior, but this provides deterministic
        random initialization and shuffling as far as practical.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # These settings improve reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# DEVICE
# =====================================================================

def get_device():
    """
    Select CUDA when available, otherwise CPU.
    """

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# =====================================================================
# GPU MEMORY
# =====================================================================

def print_gpu_memory(prefix=""):
    """
    Print current GPU memory usage.

    Useful for monitoring the RTX 3050 4 GB VRAM during local training.
    """

    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    print(
        f"{prefix}GPU memory - "
        f"allocated: {allocated:.2f} GB | "
        f"reserved: {reserved:.2f} GB"
    )


# =====================================================================
# CLIENT PARTITION VALIDATION
# =====================================================================

def validate_client_configuration():
    """
    Validate basic federated client configuration before training.

    This does not modify the partition. It only checks that the
    expected client configuration exists.
    """

    required_attributes = [
        "NUM_CLIENTS",
        "NUM_CLASSES",
        "NUM_LOCAL_EPOCHS",
        "HANDCRAFTED_FEATURE_DIM",
        "LEARNING_RATE",
        "WEIGHT_DECAY",
    ]

    missing = []

    for attribute in required_attributes:
        if not hasattr(config, attribute):
            missing.append(attribute)

    if missing:
        raise AttributeError(
            "Missing configuration values in config.py:\n"
            + "\n".join(f"  - {item}" for item in missing)
        )

    if config.NUM_CLIENTS <= 0:
        raise ValueError(
            "NUM_CLIENTS must be greater than zero."
        )

    if config.NUM_LOCAL_EPOCHS <= 0:
        raise ValueError(
            "NUM_LOCAL_EPOCHS must be greater than zero."
        )

    if config.NUM_CLASSES <= 1:
        raise ValueError(
            "NUM_CLASSES must be greater than one."
        )

    if config.HANDCRAFTED_FEATURE_DIM <= 0:
        raise ValueError(
            "HANDCRAFTED_FEATURE_DIM must be greater than zero."
        )

    if config.LEARNING_RATE <= 0:
        raise ValueError(
            "LEARNING_RATE must be greater than zero."
        )

    if config.WEIGHT_DECAY < 0:
        raise ValueError(
            "WEIGHT_DECAY cannot be negative."
        )

    print("Configuration validation: PASSED")


# =====================================================================
# BUILD LOCAL MODEL
# =====================================================================

def build_local_model(
    global_state_dict: dict,
    device: torch.device
) -> HybridClassifier:
    """
    Create a fresh local HybridClassifier and load the current global
    model weights.

    pretrained=False is intentional.

    The global model already contains the correct starting weights.
    Reinitializing ImageNet weights here would be incorrect and
    unnecessary.
    """

    model = HybridClassifier(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=config.HANDCRAFTED_FEATURE_DIM,
        pretrained=False,
    ).to(device)

    model.load_state_dict(
        global_state_dict,
        strict=True
    )

    return model


# =====================================================================
# VERIFY MODEL STATE
# =====================================================================

def verify_state_dict_compatibility(
    reference_state_dict: dict,
    candidate_state_dict: dict,
    client_id=None
):
    """
    Verify that two model state dictionaries have identical keys and
    tensor shapes.

    Required before FedAvg aggregation.
    """

    prefix = ""

    if client_id is not None:
        prefix = f"Client {client_id}: "

    reference_keys = set(
        reference_state_dict.keys()
    )

    candidate_keys = set(
        candidate_state_dict.keys()
    )

    if reference_keys != candidate_keys:

        missing = reference_keys - candidate_keys
        extra = candidate_keys - reference_keys

        raise RuntimeError(
            f"{prefix}state_dict key mismatch.\n"
            f"Missing keys: {missing}\n"
            f"Extra keys: {extra}"
        )

    for key in reference_state_dict:

        reference_shape = (
            reference_state_dict[key].shape
        )

        candidate_shape = (
            candidate_state_dict[key].shape
        )

        if reference_shape != candidate_shape:

            raise RuntimeError(
                f"{prefix}shape mismatch for '{key}': "
                f"expected {reference_shape}, "
                f"got {candidate_shape}"
            )


# =====================================================================
# CHECK WHETHER MODEL CHANGED
# =====================================================================

def calculate_parameter_change(
    before_state_dict: dict,
    after_state_dict: dict
):
    """
    Calculate an approximate L2 magnitude of the parameter update.

    This is a sanity check to ensure local training actually changed
    the model.
    """

    total_squared_change = 0.0

    for key in before_state_dict:

        before = before_state_dict[key].float()
        after = after_state_dict[key].float()

        difference = after - before

        total_squared_change += (
            torch.sum(difference * difference).item()
        )

    return total_squared_change ** 0.5


# =====================================================================
# TRAIN ONE CLIENT
# =====================================================================

def train_one_local_epoch_fedprox(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    global_state_dict,
    mu=FEDPROX_MU,
    use_amp=True,
    grad_clip=GRADIENT_CLIP,
):
    """Train one local epoch with the FedProx objective.

    Objective:
        L_i(w) + (mu / 2) * ||w - w_global||^2

    The global reference is fixed at the beginning of the federated round.
    The proximal term is evaluated over trainable model parameters only.
    """
    model.train()

    global_params = {}
    if mu > 0.0:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name in global_state_dict:
                global_params[name] = (
                    global_state_dict[name]
                    .detach()
                    .to(device=device, dtype=parameter.dtype)
                    .clone()
                )

        if not global_params:
            raise RuntimeError(
                "FedProx requires at least one trainable parameter present "
                "in global_state_dict."
            )

    total_loss = 0.0
    total_local_loss = 0.0
    total_proximal_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="    local train [FedProx]", leave=False)

    for images, features, labels in progress:
        images = images.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(images, features)
                local_loss = criterion(logits, labels)

                proximal_term = torch.zeros(
                    (), device=device, dtype=local_loss.dtype
                )
                if global_params:
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad and name in global_params:
                            proximal_term = proximal_term + torch.sum(
                                (parameter - global_params[name]) ** 2
                            )

                fedprox_loss = local_loss + (mu / 2.0) * proximal_term

            scaler.scale(fedprox_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, features)
            local_loss = criterion(logits, labels)

            proximal_term = torch.zeros(
                (), device=device, dtype=local_loss.dtype
            )
            if global_params:
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and name in global_params:
                        proximal_term = proximal_term + torch.sum(
                            (parameter - global_params[name]) ** 2
                        )

            fedprox_loss = local_loss + (mu / 2.0) * proximal_term
            fedprox_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            )
            optimizer.step()

        batch_size = images.size(0)
        total_loss += float(fedprox_loss.detach().item()) * batch_size
        total_local_loss += float(local_loss.detach().item()) * batch_size
        total_proximal_loss += float(proximal_term.detach().item()) * batch_size

        predictions = torch.argmax(logits, dim=1)
        correct += int((predictions == labels).sum().item())
        total += batch_size

        progress.set_postfix(
            loss=f"{fedprox_loss.detach().item():.4f}",
            local=f"{local_loss.detach().item():.4f}",
            acc=f"{correct / total:.4f}",
        )

    if total == 0:
        raise ValueError("Client DataLoader contains zero samples.")

    return {
        "loss": total_loss / total,
        "local_loss": total_local_loss / total,
        "proximal_term": total_proximal_loss / total,
        "accuracy": correct / total,
    }


def train_client(
    client_id: int,
    global_state_dict: dict,
    device: torch.device,
    criterion: torch.nn.Module,
    num_local_epochs: int = None,
    learning_rate: float = None,
    weight_decay: float = None,
    use_amp: bool = None,
    fedprox_mu: float = FEDPROX_MU,
):
    """Train ONE MobileNetV3-Small client for ONE FedProx round."""

    if not isinstance(client_id, int):
        raise TypeError(f"client_id must be int, got {type(client_id)}")
    if not (0 <= client_id < config.NUM_CLIENTS):
        raise ValueError(
            f"Invalid client_id={client_id}. "
            f"Expected 0 to {config.NUM_CLIENTS - 1}."
        )

    if num_local_epochs is None:
        num_local_epochs = config.NUM_LOCAL_EPOCHS
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if weight_decay is None:
        weight_decay = config.WEIGHT_DECAY
    if use_amp is None:
        use_amp = getattr(config, "USE_AMP", True) and torch.cuda.is_available()

    if num_local_epochs <= 0:
        raise ValueError("num_local_epochs must be greater than zero.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    if fedprox_mu < 0:
        raise ValueError("fedprox_mu must be >= 0.")

    start_time = time.perf_counter()

    print("\n" + "=" * 70)
    print(f"CLIENT {client_id} LOCAL TRAINING - MobileNetV3 FedProx")
    print("=" * 70)
    print(f"FedProx mu          : {fedprox_mu:.6f}")
    print(f"Gradient clip       : {GRADIENT_CLIP:.4f}")

    model = build_local_model(
        global_state_dict=global_state_dict,
        device=device,
    )

    starting_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in global_state_dict.items()
    }

    client_loader = build_client_loader(client_id=client_id)
    num_samples = len(client_loader.dataset)
    if num_samples == 0:
        raise ValueError(
            f"Client {client_id} has zero local samples.\n"
            "Check the client partition files."
        )

    # Preserve the MobileNet FedAvg reference optimizer: Adam.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    local_epoch_metrics = []

    print(
        f"\nClient {client_id}: {num_samples} samples | "
        f"{num_local_epochs} local epochs"
    )

    for local_epoch in range(1, num_local_epochs + 1):
        epoch_start = time.perf_counter()

        metrics = train_one_local_epoch_fedprox(
            model=model,
            loader=client_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            global_state_dict=global_state_dict,
            mu=fedprox_mu,
            use_amp=use_amp,
            grad_clip=GRADIENT_CLIP,
        )

        epoch_time = time.perf_counter() - epoch_start
        local_epoch_metrics.append(
            {
                "local_epoch": local_epoch,
                "train_loss": float(metrics["loss"]),
                "local_loss": float(metrics["local_loss"]),
                "proximal_term": float(metrics["proximal_term"]),
                "train_accuracy": float(metrics["accuracy"]),
                "epoch_time_sec": float(epoch_time),
            }
        )

        print(
            f"  Local epoch {local_epoch}/{num_local_epochs} | "
            f"FedProx loss: {metrics['loss']:.4f} | "
            f"CE loss: {metrics['local_loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prox: {metrics['proximal_term']:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

    cpu_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    verify_state_dict_compatibility(
        reference_state_dict=starting_state_dict,
        candidate_state_dict=cpu_state_dict,
        client_id=client_id,
    )

    parameter_update_norm = calculate_parameter_change(
        before_state_dict=starting_state_dict,
        after_state_dict=cpu_state_dict,
    )

    if parameter_update_norm <= 0.0:
        raise RuntimeError(
            f"Client {client_id}: local FedProx training did not change "
            "the model parameters."
        )

    final_train_loss = local_epoch_metrics[-1]["train_loss"]
    final_train_acc = local_epoch_metrics[-1]["train_accuracy"]
    total_time = time.perf_counter() - start_time

    del model
    del optimizer
    del scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "client_id": client_id,
        "state_dict": cpu_state_dict,
        "num_samples": num_samples,
        "local_epoch_metrics": local_epoch_metrics,
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "parameter_update_norm": parameter_update_norm,
        "fedprox_mu": float(fedprox_mu),
        "local_epoch_time_sec": total_time / num_local_epochs,
        "total_training_time_sec": total_time,
    }


# =====================================================================
# BUILD INITIAL GLOBAL MODEL
# =====================================================================

def build_initial_global_state(device):
    """
    Build the initial global model.

    For Round 1 this is an ImageNet-pretrained MobileNetV3-Small
    backbone combined with the handcrafted-feature classification head.

    Only the state_dict is retained after initialization.
    """

    print(
        "\nBuilding initial global model "
        "(ImageNet-pretrained)..."
    )

    global_model = HybridClassifier(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=config.HANDCRAFTED_FEATURE_DIM,
        pretrained=True,
    ).to(device)

    global_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    parameter_count = sum(
        parameter.numel()
        for parameter in global_model.parameters()
    )

    print(
        f"Initial global model parameters: "
        f"{parameter_count:,}"
    )

    del global_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return global_state_dict


# =====================================================================
# BUILD SHARED LOSS
# =====================================================================

def build_shared_criterion(device):
    """
    Build the same loss function used by the centralized baseline.

    Class weights are computed from the GLOBAL centralized training
    distribution, not separately for each client.
    """

    label_smoothing = getattr(
        config,
        "LABEL_SMOOTHING",
        0.05
    )

    print(
        "\nComputing globally-shared class weights..."
    )

    class_weights = compute_class_weights(
        device
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )

    print(
        f"Using weighted CrossEntropyLoss "
        f"(label_smoothing={label_smoothing})."
    )

    return criterion


# =====================================================================
# SELF-TEST
# =====================================================================

def main():

    print("=" * 70)
    print(
        "PHASE 2 - STEP 9: "
        "FEDERATED CLIENT LOGIC SELF-TEST"
    )
    print("=" * 70)

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------

    seed = getattr(
        config,
        "RANDOM_SEED",
        42
    )

    set_seed(seed)

    print(
        f"\nRandom seed: {seed}"
    )

    # -----------------------------------------------------------------
    # Validate configuration
    # -----------------------------------------------------------------

    validate_client_configuration()

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU VRAM: "
            f"{total_memory:.2f} GB"
        )

    # -----------------------------------------------------------------
    # Training configuration
    # -----------------------------------------------------------------

    use_amp = (
        getattr(config, "USE_AMP", True)
        and torch.cuda.is_available()
    )

    label_smoothing = getattr(
        config,
        "LABEL_SMOOTHING",
        0.05
    )

    print(
        "\nFederated configuration:"
    )

    print(
        f"  Number of clients       : "
        f"{config.NUM_CLIENTS}"
    )

    print(
        f"  Local epochs            : "
        f"{config.NUM_LOCAL_EPOCHS}"
    )

    print(
        f"  Learning rate           : "
        f"{config.LEARNING_RATE}"
    )

    print(
        f"  Weight decay            : "
        f"{config.WEIGHT_DECAY}"
    )

    print(
        f"  Label smoothing         : "
        f"{label_smoothing}"
    )

    print(
        f"  FedProx mu              : {FEDPROX_MU}"
    )

    print(
        f"  Gradient clipping       : {GRADIENT_CLIP}"
    )

    print(
        f"  AMP                     : "
        f"{use_amp}"
    )

    print(
        f"  Handcrafted features   : "
        f"{config.HANDCRAFTED_FEATURE_DIM}"
    )

    # -----------------------------------------------------------------
    # Initial global model
    # -----------------------------------------------------------------

    global_state_dict = build_initial_global_state(
        device
    )

    # -----------------------------------------------------------------
    # Shared loss
    # -----------------------------------------------------------------

    criterion = build_shared_criterion(
        device
    )

    # -----------------------------------------------------------------
    # Validate that every client can be trained
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "TESTING LOCAL TRAINING ON EVERY CLIENT"
    )
    print("=" * 70)

    print(
        "\nNOTE:"
    )

    print(
        "The self-test uses 1 local epoch per client "
        "to verify correctness quickly."
    )

    print(
        "The real federated experiment will use "
        f"{config.NUM_LOCAL_EPOCHS} local epochs."
    )

    results = []

    total_start = time.perf_counter()

    # -----------------------------------------------------------------
    # Train each client once
    # -----------------------------------------------------------------

    for client_id in range(
        config.NUM_CLIENTS
    ):

        print("\n" + "-" * 70)
        print(
            f"CLIENT {client_id}"
        )
        print("-" * 70)

        # -------------------------------------------------------------
        # Verify every client starts from exactly the same global model
        # -------------------------------------------------------------

        client_start_state = {
            key: value.detach().cpu().clone()
            for key, value in global_state_dict.items()
        }

        result = train_client(
            client_id=client_id,

            global_state_dict=global_state_dict,

            device=device,

            criterion=criterion,

            # Self-test: only 1 local epoch.
            num_local_epochs=1,
        )

        # -------------------------------------------------------------
        # Validate result
        # -------------------------------------------------------------

        assert (
            result["client_id"] == client_id
        ), (
            f"Incorrect client ID returned: "
            f"{result['client_id']}"
        )

        assert (
            result["num_samples"] > 0
        ), (
            f"Client {client_id} "
            f"returned zero samples."
        )

        verify_state_dict_compatibility(
            reference_state_dict=client_start_state,
            candidate_state_dict=result["state_dict"],
            client_id=client_id
        )

        # -------------------------------------------------------------
        # Check that training actually changed parameters
        # -------------------------------------------------------------

        update_norm = (
            result["parameter_update_norm"]
        )

        if update_norm == 0.0:

            raise RuntimeError(
                f"Client {client_id}: "
                f"model parameters did not change "
                f"after local training."
            )

        print(
            f"\n  Samples used          : "
            f"{result['num_samples']}"
        )

        print(
            f"  Final local loss      : "
            f"{result['final_train_loss']:.4f}"
        )

        print(
            f"  Final local accuracy  : "
            f"{result['final_train_acc']:.4f}"
        )

        print(
            f"  Parameter update norm : "
            f"{update_norm:.6f}"
        )

        print(
            f"  Total local time      : "
            f"{result['total_training_time_sec']:.1f}s"
        )

        results.append(result)

        # Free any cached CUDA memory before next client.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Final validation
    # -----------------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    total_samples = sum(
        result["num_samples"]
        for result in results
    )

    expected_total = 8112

    print("\n" + "=" * 70)
    print(
        "FINAL STEP 9 VALIDATION"
    )
    print("=" * 70)

    print(
        f"\nNumber of clients tested : "
        f"{len(results)}"
    )

    print(
        f"Total client samples    : "
        f"{total_samples}"
    )

    # The current Phase 1 training split contains 8112 samples.
    if total_samples != expected_total:

        print(
            "\nWARNING:"
        )

        print(
            f"Expected {expected_total} samples "
            f"from the current training split, "
            f"but found {total_samples}."
        )

    else:

        print(
            "Sample conservation check: PASSED"
        )

    # -----------------------------------------------------------------
    # Print client sample distribution
    # -----------------------------------------------------------------

    print(
        "\nClient sample distribution:"
    )

    for result in results:

        print(
            f"  Client {result['client_id']}: "
            f"{result['num_samples']} samples"
        )

    # -----------------------------------------------------------------
    # State dictionary validation
    # -----------------------------------------------------------------

    for result in results:

        verify_state_dict_compatibility(
            reference_state_dict=global_state_dict,
            candidate_state_dict=result["state_dict"],
            client_id=result["client_id"]
        )

    print(
        "\nState_dict compatibility check: PASSED"
    )

    # -----------------------------------------------------------------
    # Parameter update validation
    # -----------------------------------------------------------------

    update_norms = [
        result["parameter_update_norm"]
        for result in results
    ]

    print(
        "\nParameter update sanity check: PASSED"
    )

    print(
        f"  Minimum update norm: "
        f"{min(update_norms):.6f}"
    )

    print(
        f"  Maximum update norm: "
        f"{max(update_norms):.6f}"
    )

    # -----------------------------------------------------------------
    # GPU memory
    # -----------------------------------------------------------------

    if torch.cuda.is_available():

        print(
            "\nFinal GPU memory:"
        )

        print_gpu_memory(
            prefix="  "
        )

    # -----------------------------------------------------------------
    # Completion
    # -----------------------------------------------------------------

    print(
        "\nTotal self-test time: "
        f"{total_time / 60:.2f} minutes"
    )

    print("\n" + "=" * 70)
    print(
        "PHASE 2 - STEP 9 COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nAll {config.NUM_CLIENTS} clients "
        f"trained successfully."
    )

    print(
        f"Total samples across clients: "
        f"{total_samples}"
    )

    print(
        "All client state_dicts match "
        "the global model."
    )

    print(
        "All clients produced non-zero "
        "parameter updates."
    )

    print(
        "\nThe client-side local training logic "
        "is ready for FedAvg."
    )

    print(
        "\nNext:"
    )

    print(
        "  10_fedavg_server.py"
    )

    print(
        "  -> sample-weighted FedAvg aggregation"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()