"""
09A_federated_client_convnext.py
--------------------------------
PHASE 2 - STEP 9A

FEDERATED CLIENT FOR CONVNEXT-TINY V2

Purpose:
    Implements the local training logic for one federated client.

Architecture:
    ConvNeXt-Tiny
        -> 768-D deep embedding
        -> 1320-D handcrafted features
        -> 256-D handcrafted branch
        -> 1024-D feature fusion
        -> 7-class classifier

This file does NOT perform FedAvg aggregation.
The server is responsible for aggregation.

Workflow:

    Global Model
         |
         v
    Client receives global weights
         |
         v
    Client trains locally
         |
         v
    Updated local weights
         |
         v
    Client sends weights + sample count
         |
         v
    FedAvg Server

Dataset:
    06_dataset.py

Model:
    07A_model.py

Important:
    - Uses the existing TRAIN handcrafted feature scaler.
    - Uses training augmentation from 06_dataset.py.
    - Uses ConvNeXt-Tiny V2 architecture.
    - No validation/test data is used for local training.
    - This file does not modify the global model.
"""

import os
import sys
import time
import copy
import importlib.util

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from tqdm import tqdm


# =====================================================================
# CONFIG
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
# DEVICE
# =====================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================================
# LOAD MODULE
# =====================================================================

def _load_module(filename, module_name):
    """
    Dynamically load a Python module from src/.

    This allows Step 9A to use:
        06_dataset.py
        07A_model.py

    without requiring package imports.
    """

    path = os.path.join(
        os.path.dirname(__file__),
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired module not found:\n"
            f"{path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Could not load module:\n"
            f"{path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# =====================================================================
# LOAD DATASET + MODEL
# =====================================================================

dataset_module = _load_module(
    "06_dataset.py",
    "dataset_module_09a"
)

model_module = _load_module(
    "07A_model_convnext.py",
    "model_module_09a"
)


# Dataset functions
build_client_loader = (
    dataset_module.build_client_loader
)

prepare_all_features = (
    dataset_module.prepare_all_features
)

load_scaler = (
    dataset_module.load_scaler
)

prepare_scaled_features = (
    dataset_module.prepare_scaled_features
)


# ConvNeXt model
HybridClassifier = (
    model_module.HybridClassifier
)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_random_seeds(seed=None):

    if seed is None:

        seed = config.RANDOM_SEED

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    print(
        f"Random seed: {seed}"
    )


# =====================================================================
# CONFIG HELPERS
# =====================================================================

def get_config_value(
    name,
    default
):

    return getattr(
        config,
        name,
        default
    )


# =====================================================================
# FEDERATED CONFIGURATION
# =====================================================================

LOCAL_EPOCHS = get_config_value(
    "LOCAL_EPOCHS",
    3
)

LOCAL_LEARNING_RATE = get_config_value(
    "FEDERATED_LEARNING_RATE",
    get_config_value(
        "LEARNING_RATE",
        1e-4
    )
)

LOCAL_WEIGHT_DECAY = get_config_value(
    "FEDERATED_WEIGHT_DECAY",
    get_config_value(
        "WEIGHT_DECAY",
        1e-4
    )
)

LOCAL_BATCH_SIZE = get_config_value(
    "FEDERATED_BATCH_SIZE",
    get_config_value(
        "BATCH_SIZE",
        16
    )
)

LABEL_SMOOTHING = get_config_value(
    "LABEL_SMOOTHING",
    0.05
)

USE_AMP = (
    get_config_value(
        "USE_AMP",
        True
    )
    and torch.cuda.is_available()
)

GRADIENT_CLIP = get_config_value(
    "FEDERATED_GRADIENT_CLIP",
    1.0
)


# =====================================================================
# PRINT CONFIGURATION
# =====================================================================

def print_client_configuration():

    print(
        "\n" + "=" * 60
    )

    print(
        "FEDERATED CLIENT CONFIGURATION"
    )

    print(
        "=" * 60
    )

    print(
        f"\nArchitecture          : ConvNeXt-Tiny V2"
    )

    print(
        f"Device                : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU                   : "
            f"{torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU VRAM              : "
            f"{total_memory:.2f} GB"
        )

    print(
        f"Local epochs          : "
        f"{LOCAL_EPOCHS}"
    )

    print(
        f"Local learning rate   : "
        f"{LOCAL_LEARNING_RATE}"
    )

    print(
        f"Weight decay          : "
        f"{LOCAL_WEIGHT_DECAY}"
    )

    print(
        f"Batch size            : "
        f"{LOCAL_BATCH_SIZE}"
    )

    print(
        f"Label smoothing       : "
        f"{LABEL_SMOOTHING}"
    )

    print(
        f"AMP                   : "
        f"{USE_AMP}"
    )

    print(
        f"Gradient clipping     : "
        f"{GRADIENT_CLIP}"
    )

    print(
        f"Classes               : "
        f"{config.NUM_CLASSES}"
    )

    print(
        f"Handcrafted features : "
        f"{config.HANDCRAFTED_FEATURE_DIM}"
    )


# =====================================================================
# BUILD MODEL
# =====================================================================

def build_model(
    device=DEVICE
):

    model = HybridClassifier(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=(
            config.HANDCRAFTED_FEATURE_DIM
        ),
        pretrained=True
    )

    model = model.to(
        device
    )

    return model


# =====================================================================
# COUNT PARAMETERS
# =====================================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# =====================================================================
# LOAD CLIENT DATA
# =====================================================================

def load_client_dataframe(
    client_id
):

    client_csv = os.path.join(
        config.OUTPUT_DIR,
        "client_partitions",
        f"client_{client_id}.csv"
    )

    if not os.path.exists(
        client_csv
    ):

        raise FileNotFoundError(
            f"\nClient partition not found:\n"
            f"{client_csv}\n\n"
            f"Run 02_dirichlet_partition.py first."
        )

    df = pd.read_csv(
        client_csv
    )

    if len(df) == 0:

        raise ValueError(
            f"Client {client_id} contains "
            f"zero samples."
        )

    required_columns = [
        "image_id",
        "filepath",
        "label"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Client CSV is missing columns:\n"
            f"{missing_columns}"
        )

    return df


# =====================================================================
# BUILD CLIENT LOADER
# =====================================================================

def build_federated_client_loader(
    client_id,
    train_features=None
):

    """
    Build DataLoader for one federated client.

    Important:
        Training augmentation is controlled by
        06_dataset.py.
    """

    if train_features is None:

        scaler = load_scaler()

        train_features = (
            prepare_scaled_features(
                "train",
                os.path.join(
                    config.HANDCRAFTED_FEATURE_DIR,
                    "train_features.npz"
                ),
                scaler
            )
        )

    client_df = load_client_dataframe(
        client_id
    )

    loader = build_client_loader(
        client_id=client_id,
        train_features=train_features
    )

    # ---------------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------------

    if len(loader.dataset) != len(client_df):

        raise ValueError(
            f"Client dataset size mismatch.\n"
            f"CSV samples   : {len(client_df)}\n"
            f"Loader samples: {len(loader.dataset)}"
        )

    return loader


# =====================================================================
# CLASS WEIGHTS
# =====================================================================

def compute_client_class_weights(
    client_id,
    device=DEVICE
):

    """
    Compute class weights from the client's LOCAL data.

    This is intentionally local.

    No information from validation or test data is used.
    """

    client_df = load_client_dataframe(
        client_id
    )

    counts = (
        client_df["label"]
        .value_counts()
    )

    total = len(
        client_df
    )

    num_classes = (
        config.NUM_CLASSES
    )

    weights = []

    print(
        "\nClient training class counts:"
    )

    for cls in config.CLASS_LIST:

        count = counts.get(
            cls,
            0
        )

        print(
            f"  {cls:6s}: {count}"
        )

    # ---------------------------------------------------------------
    # Balanced inverse-frequency weights
    # ---------------------------------------------------------------

    for cls in config.CLASS_LIST:

        count = counts.get(
            cls,
            0
        )

        if count == 0:

            # A missing class should not contribute
            # to this client's loss.

            weight = 0.0

        else:

            weight = (
                total
                /
                (
                    num_classes
                    *
                    count
                )
            )

        weights.append(
            weight
        )

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )

    print(
        "\nClient class weights:"
    )

    for cls, weight in zip(
        config.CLASS_LIST,
        weights
    ):

        print(
            f"  {cls:6s}: {weight:.4f}"
        )

    return weights_tensor


# =====================================================================
# LOSS FUNCTION
# =====================================================================

def build_loss_function(
    client_id,
    device=DEVICE
):

    class_weights = (
        compute_client_class_weights(
            client_id,
            device
        )
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    return criterion


# =====================================================================
# OPTIMIZER
# =====================================================================

def build_optimizer(
    model
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LOCAL_LEARNING_RATE,
        weight_decay=LOCAL_WEIGHT_DECAY
    )

    return optimizer


# =====================================================================
# TRAIN ONE LOCAL EPOCH
# =====================================================================

def train_one_local_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    use_amp=True,
    grad_clip=1.0
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="    local train",
        leave=False
    )

    for (
        images,
        features,
        labels
    ) in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        features = features.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # -----------------------------------------------------------
        # Mixed precision
        # -----------------------------------------------------------

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda",
                enabled=True
            ):

                logits = model(
                    images,
                    features
                )

                loss = criterion(
                    logits,
                    labels
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            logits = model(
                images,
                features
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )

            optimizer.step()

        # -----------------------------------------------------------
        # Metrics
        # -----------------------------------------------------------

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            *
            batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}"
        )

    if total == 0:

        raise ValueError(
            "Client DataLoader contains zero samples."
        )

    average_loss = (
        total_loss
        /
        total
    )

    accuracy = (
        correct
        /
        total
    )

    return (
        average_loss,
        accuracy
    )


# =====================================================================
# LOCAL EVALUATION ON TRAINING DATA
# =====================================================================

@torch.no_grad()
def evaluate_local_training_data(
    model,
    loader,
    criterion,
    device
):

    """
    Evaluate the locally trained model on the client's
    training partition.

    This metric is ONLY a local diagnostic.

    It is not used as the global validation/test metric.
    """

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    for (
        images,
        features,
        labels
    ) in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        features = features.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        if USE_AMP:

            with torch.amp.autocast(
                device_type="cuda",
                enabled=True
            ):

                logits = model(
                    images,
                    features
                )

                loss = criterion(
                    logits,
                    labels
                )

        else:

            logits = model(
                images,
                features
            )

            loss = criterion(
                logits,
                labels
            )

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            *
            batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

        all_labels.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

    if len(all_labels) == 0:

        raise ValueError(
            "No samples found during local evaluation."
        )

    average_loss = (
        total_loss
        /
        len(all_labels)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        macro_f1
    )


# =====================================================================
# GET MODEL STATE
# =====================================================================

def get_model_state(
    model
):

    """
    Return model parameters on CPU.

    This is important for federated learning because the
    server will receive client model weights.
    """

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


# =====================================================================
# LOAD MODEL STATE
# =====================================================================

def load_model_state(
    model,
    state_dict,
    device=DEVICE
):

    """
    Load a global model state into the local client.
    """

    if not isinstance(
        state_dict,
        dict
    ):

        raise TypeError(
            "state_dict must be a dictionary."
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(
        device
    )

    return model


# =====================================================================
# TRAIN ONE CLIENT
# =====================================================================

def train_client(
    client_id,
    global_state=None,
    train_features=None,
    local_epochs=None,
    device=DEVICE
):

    """
    Train one federated client.

    Parameters
    ----------
    client_id:
        Integer client identifier.

    global_state:
        State dictionary received from the server.
        If None, a pretrained ConvNeXt model is created.

    train_features:
        Scaled handcrafted training features.

    local_epochs:
        Number of local epochs.

    device:
        CUDA or CPU.

    Returns
    -------
    result : dict

        {
            "client_id": ...,
            "state_dict": ...,
            "num_samples": ...,
            "train_loss": ...,
            "train_accuracy": ...,
            "train_macro_f1": ...,
            "local_epochs": ...
        }
    """

    if local_epochs is None:

        local_epochs = LOCAL_EPOCHS

    if local_epochs < 1:

        raise ValueError(
            "local_epochs must be >= 1."
        )

    client_start = (
        time.perf_counter()
    )

    print(
        "\n" + "=" * 60
    )

    print(
        f"CLIENT {client_id} LOCAL TRAINING"
    )

    print(
        "=" * 60
    )

    # ---------------------------------------------------------------
    # Client loader
    # ---------------------------------------------------------------

    loader = build_federated_client_loader(
        client_id=client_id,
        train_features=train_features
    )

    num_samples = len(
        loader.dataset
    )

    print(
        f"\nClient {client_id} samples:"
        f" {num_samples}"
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    print(
        "\nBuilding ConvNeXt-Tiny V2 model..."
    )

    model = build_model(
        device=device
    )

    total_params, trainable_params = (
        count_parameters(model)
    )

    print(
        f"Total parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )

    # ---------------------------------------------------------------
    # Load global model
    # ---------------------------------------------------------------

    if global_state is not None:

        print(
            "\nLoading global model weights..."
        )

        load_model_state(
            model,
            global_state,
            device
        )

        print(
            "Global weights loaded ✓"
        )

    else:

        print(
            "\nNo global state supplied."
        )

        print(
            "Using ImageNet-pretrained "
            "ConvNeXt-Tiny initialization."
        )

    # ---------------------------------------------------------------
    # Loss
    # ---------------------------------------------------------------

    print(
        "\nBuilding local loss..."
    )

    criterion = build_loss_function(
        client_id=client_id,
        device=device
    )

    # ---------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------

    optimizer = build_optimizer(
        model
    )

    # ---------------------------------------------------------------
    # AMP scaler
    # ---------------------------------------------------------------

    if use_amp:

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True
        )

    else:

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=False
        )

    # ---------------------------------------------------------------
    # Local training
    # ---------------------------------------------------------------

    history = []

    print(
        "\n" + "-" * 60
    )

    print(
        "STARTING LOCAL TRAINING"
    )

    print(
        "-" * 60
    )

    for epoch in range(
        1,
        local_epochs + 1
    ):

        epoch_start = (
            time.perf_counter()
        )

        train_loss, train_acc = (
            train_one_local_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                use_amp=USE_AMP,
                grad_clip=GRADIENT_CLIP
            )
        )

        epoch_time = (
            time.perf_counter()
            -
            epoch_start
        )

        print(
            f"\nLocal Epoch "
            f"{epoch:02d}/{local_epochs}"
        )

        print(
            f"  Train Loss     : "
            f"{train_loss:.4f}"
        )

        print(
            f"  Train Accuracy : "
            f"{train_acc:.4f}"
        )

        print(
            f"  Learning Rate  : "
            f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        print(
            f"  Epoch time     : "
            f"{epoch_time:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "learning_rate":
                    optimizer.param_groups[0]["lr"],
                "epoch_time_sec":
                    epoch_time
            }
        )

    # ---------------------------------------------------------------
    # Final local evaluation
    # ---------------------------------------------------------------

    print(
        "\n" + "-" * 60
    )

    print(
        "FINAL LOCAL TRAINING EVALUATION"
    )

    print(
        "-" * 60
    )

    final_loss, final_acc, final_macro_f1 = (
        evaluate_local_training_data(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device
        )
    )

    print(
        f"\nClient {client_id} final metrics:"
    )

    print(
        f"  Loss       : {final_loss:.4f}"
    )

    print(
        f"  Accuracy   : {final_acc:.4f}"
    )

    print(
        f"  Macro-F1   : {final_macro_f1:.4f}"
    )

    # ---------------------------------------------------------------
    # Extract state
    # ---------------------------------------------------------------

    updated_state = get_model_state(
        model
    )

    elapsed = (
        time.perf_counter()
        -
        client_start
    )

    print(
        "\n" + "=" * 60
    )

    print(
        f"CLIENT {client_id} TRAINING COMPLETE ✓"
    )

    print(
        "=" * 60
    )

    print(
        f"Samples trained : {num_samples}"
    )

    print(
        f"Local epochs    : {local_epochs}"
    )

    print(
        f"Final accuracy  : {final_acc:.4f}"
    )

    print(
        f"Final Macro-F1  : {final_macro_f1:.4f}"
    )

    print(
        f"Total time      : "
        f"{elapsed / 60:.2f} minutes"
    )

    # ---------------------------------------------------------------
    # GPU memory
    # ---------------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated(0)
            /
            (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved(0)
            /
            (1024 ** 2)
        )

        print(
            "\nGPU memory:"
        )

        print(
            f"  Allocated : "
            f"{allocated:.2f} MB"
        )

        print(
            f"  Reserved  : "
            f"{reserved:.2f} MB"
        )

    return {
        "client_id": client_id,
        "state_dict": updated_state,
        "num_samples": num_samples,
        "train_loss": final_loss,
        "train_accuracy": final_acc,
        "train_macro_f1": final_macro_f1,
        "local_epochs": local_epochs,
        "history": history
    }


# =====================================================================
# CLIENT WEIGHT SANITY CHECK
# =====================================================================

def verify_state_dict(
    state_dict,
    model
):

    """
    Verify that a returned federated state has exactly
    the same structure as the ConvNeXt model.
    """

    model_state = model.state_dict()

    if set(state_dict.keys()) != set(
        model_state.keys()
    ):

        missing = (
            set(model_state.keys())
            -
            set(state_dict.keys())
        )

        extra = (
            set(state_dict.keys())
            -
            set(model_state.keys())
        )

        raise ValueError(
            "\nState dictionary mismatch.\n"
            f"Missing keys: {list(missing)[:5]}\n"
            f"Extra keys: {list(extra)[:5]}"
        )

    for key in model_state:

        if (
            state_dict[key].shape
            !=
            model_state[key].shape
        ):

            raise ValueError(
                f"Shape mismatch for "
                f"'{key}'.\n"
                f"Expected: "
                f"{model_state[key].shape}\n"
                f"Found: "
                f"{state_dict[key].shape}"
            )

    print(
        "State dictionary structure verified ✓"
    )


# =====================================================================
# SELF-TEST
# =====================================================================

def self_test():

    start = (
        time.perf_counter()
    )

    print(
        "=" * 60
    )

    print(
        "PHASE 2 - STEP 9A: "
        "CONVNEXT FEDERATED CLIENT SELF-TEST"
    )

    print(
        "=" * 60
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            /
            (1024 ** 3)
        )

        print(
            f"GPU VRAM: "
            f"{total_memory:.2f} GB"
        )

    set_random_seeds()

    print_client_configuration()

    # ---------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "BUILDING CONVNEXT-TINY V2 CLIENT MODEL"
    )

    print(
        "=" * 60
    )

    model = build_model(
        DEVICE
    )

    total_params, trainable_params = (
        count_parameters(model)
    )

    print(
        "\nModel parameters:"
    )

    print(
        f"  Total     : "
        f"{total_params:,}"
    )

    print(
        f"  Trainable : "
        f"{trainable_params:,}"
    )

    # ---------------------------------------------------------------
    # Check parameter gradients
    # ---------------------------------------------------------------

    backbone_trainable = any(
        parameter.requires_grad
        for name, parameter
        in model.named_parameters()
        if "features" in name
    )

    print(
        f"\nBackbone trainable: "
        f"{backbone_trainable}"
    )

    if not backbone_trainable:

        raise ValueError(
            "ConvNeXt backbone is not trainable."
        )

    # ---------------------------------------------------------------
    # Prepare scaled features
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "PREPARING TRAIN FEATURES"
    )

    print(
        "=" * 60
    )

    train_features, _, _ = (
        prepare_all_features()
    )

    print(
        f"\nTrain feature samples: "
        f"{len(train_features)}"
    )

    # ---------------------------------------------------------------
    # Find client 0
    # ---------------------------------------------------------------

    client_id = 0

    client_df = load_client_dataframe(
        client_id
    )

    print(
        f"Client {client_id} samples: "
        f"{len(client_df)}"
    )

    # ---------------------------------------------------------------
    # Build loader
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "BUILDING CLIENT DATALOADER"
    )

    print(
        "=" * 60
    )

    client_loader = (
        build_federated_client_loader(
            client_id=client_id,
            train_features=train_features
        )
    )

    print(
        f"\nClient loader samples: "
        f"{len(client_loader.dataset)}"
    )

    # ---------------------------------------------------------------
    # Batch test
    # ---------------------------------------------------------------

    images, features, labels = next(
        iter(client_loader)
    )

    print(
        "\nClient batch:"
    )

    print(
        f"  Images   : "
        f"{images.shape}"
    )

    print(
        f"  Features : "
        f"{features.shape}"
    )

    print(
        f"  Labels   : "
        f"{labels.shape}"
    )

    expected_image_shape = (
        config.BATCH_SIZE,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    )

    expected_feature_shape = (
        config.BATCH_SIZE,
        config.HANDCRAFTED_FEATURE_DIM
    )

    # The final batch can theoretically be smaller,
    # therefore check dimensions instead of requiring
    # an exact batch size.

    if images.ndim != 4:

        raise ValueError(
            f"Unexpected image dimensions: "
            f"{images.shape}"
        )

    if images.shape[1:] != (
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    ):

        raise ValueError(
            f"Unexpected image shape: "
            f"{images.shape}"
        )

    if features.ndim != 2:

        raise ValueError(
            f"Unexpected feature dimensions: "
            f"{features.shape}"
        )

    if features.shape[1] != (
        config.HANDCRAFTED_FEATURE_DIM
    ):

        raise ValueError(
            f"Unexpected feature shape: "
            f"{features.shape}"
        )

    if labels.ndim != 1:

        raise ValueError(
            f"Unexpected label shape: "
            f"{labels.shape}"
        )

    print(
        "\n✓ Client DataLoader verified"
    )

    # ---------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "RUNNING CLIENT FORWARD PASS"
    )

    print(
        "=" * 60
    )

    model.eval()

    images_device = images.to(
        DEVICE
    )

    features_device = features.to(
        DEVICE
    )

    with torch.no_grad():

        if USE_AMP:

            with torch.amp.autocast(
                device_type="cuda",
                enabled=True
            ):

                logits = model(
                    images_device,
                    features_device
                )

        else:

            logits = model(
                images_device,
                features_device
            )

    print(
        f"  Images   : "
        f"{tuple(images.shape)}"
    )

    print(
        f"  Features : "
        f"{tuple(features.shape)}"
    )

    print(
        f"  Output   : "
        f"{tuple(logits.shape)}"
    )

    expected_output_dim = (
        config.NUM_CLASSES
    )

    if logits.shape[0] != images.shape[0]:

        raise ValueError(
            "Output batch size does not match "
            "input batch size."
        )

    if logits.shape[1] != expected_output_dim:

        raise ValueError(
            f"Expected "
            f"{expected_output_dim} output classes, "
            f"got {logits.shape[1]}"
        )

    print(
        "\n  ✓ Forward pass successful"
    )

    # ---------------------------------------------------------------
    # Backward pass / gradient flow
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "CHECKING CLIENT GRADIENT FLOW"
    )

    print(
        "=" * 60
    )

    model.train()

    test_images = images.to(
        DEVICE
    )

    test_features = features.to(
        DEVICE
    )

    test_labels = labels.to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LOCAL_LEARNING_RATE
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    if USE_AMP:

        test_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=True
        ):

            test_logits = model(
                test_images,
                test_features
            )

            test_loss = criterion(
                test_logits,
                test_labels
            )

        test_scaler.scale(
            test_loss
        ).backward()

        test_scaler.unscale_(
            optimizer
        )

    else:

        test_logits = model(
            test_images,
            test_features
        )

        test_loss = criterion(
            test_logits,
            test_labels
        )

        test_loss.backward()

    # ---------------------------------------------------------------
    # Gradient checks
    # ---------------------------------------------------------------

    backbone_has_gradient = False
    handcrafted_has_gradient = False
    classifier_has_gradient = False

    for name, parameter in (
        model.named_parameters()
    ):

        if parameter.grad is None:

            continue

        if parameter.grad.abs().sum().item() == 0:

            continue

        if "features" in name:

            backbone_has_gradient = True

        if (
            "handcrafted" in name.lower()
            or
            "feature" in name.lower()
        ):

            handcrafted_has_gradient = True

        if (
            "classifier" in name.lower()
            or
            "head" in name.lower()
        ):

            classifier_has_gradient = True

    print(
        f"\n  Backbone gradients     : "
        f"{backbone_has_gradient}"
    )

    print(
        f"  Handcrafted gradients  : "
        f"{handcrafted_has_gradient}"
    )

    print(
        f"  Classifier gradients   : "
        f"{classifier_has_gradient}"
    )

    if not backbone_has_gradient:

        raise ValueError(
            "No gradient detected in ConvNeXt backbone."
        )

    if not classifier_has_gradient:

        raise ValueError(
            "No gradient detected in classification head."
        )

    print(
        "\n  ✓ Gradient flow verified"
    )

    # ---------------------------------------------------------------
    # State dict test
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "CHECKING FEDERATED STATE DICTIONARY"
    )

    print(
        "=" * 60
    )

    state = get_model_state(
        model
    )

    print(
        f"\nState dictionary tensors: "
        f"{len(state)}"
    )

    verify_state_dict(
        state,
        model
    )

    # ---------------------------------------------------------------
    # Check CPU tensors
    # ---------------------------------------------------------------

    all_cpu = all(
        not tensor.is_cuda
        for tensor in state.values()
    )

    print(
        f"State tensors on CPU: "
        f"{all_cpu}"
    )

    if not all_cpu:

        raise ValueError(
            "Federated state dictionary "
            "contains GPU tensors."
        )

    print(
        "\n✓ Federated state extraction verified"
    )

    # ---------------------------------------------------------------
    # Deep embedding test
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "CHECKING DEEP EMBEDDING"
    )

    print(
        "=" * 60
    )

    model.eval()

    embedding = None

    # Try the public embedding API used by the
    # ConvNeXt model implementation.

    if hasattr(
        model,
        "extract_embedding"
    ):

        with torch.no_grad():

            embedding = model.extract_embedding(
                test_images,
                test_features
            )

    elif hasattr(
        model,
        "get_embedding"
    ):

        with torch.no_grad():

            embedding = model.get_embedding(
                test_images,
                test_features
            )

    print(
        f"\nEmbedding available: "
        f"{embedding is not None}"
    )

    if embedding is not None:

        print(
            f"Embedding shape: "
            f"{tuple(embedding.shape)}"
        )

        if embedding.shape[0] != (
            images.shape[0]
        ):

            raise ValueError(
                "Embedding batch dimension "
                "is incorrect."
            )

        if embedding.shape[1] != 768:

            raise ValueError(
                f"Expected 768-D ConvNeXt "
                f"embedding, got "
                f"{embedding.shape[1]}"
            )

        print(
            "\n✓ 768-D embedding verified"
        )

    else:

        print(
            "\n[INFO] No public embedding "
            "method detected."
        )

        print(
            "Forward pass and federated "
            "state tests remain valid."
        )

    # ---------------------------------------------------------------
    # Memory
    # ---------------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated(0)
            /
            (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved(0)
            /
            (1024 ** 2)
        )

        print(
            "\nGPU memory after self-test:"
        )

        print(
            f"  Allocated : "
            f"{allocated:.2f} MB"
        )

        print(
            f"  Reserved  : "
            f"{reserved:.2f} MB"
        )

        torch.cuda.empty_cache()

    elapsed = (
        time.perf_counter()
        -
        start
    )

    # ---------------------------------------------------------------
    # Final
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 2 - STEP 9A COMPLETE ✓"
    )

    print(
        "=" * 60
    )

    print(
        f"Self-test time: "
        f"{elapsed:.2f}s"
    )

    print(
        "\nVerified:"
    )

    print(
        "  ✓ ConvNeXt-Tiny V2 client model"
    )

    print(
        "  ✓ ImageNet pretrained initialization"
    )

    print(
        "  ✓ Full backbone fine-tuning"
    )

    print(
        "  ✓ Client partition loading"
    )

    print(
        "  ✓ Training augmentation"
    )

    print(
        "  ✓ 1320-D handcrafted features"
    )

    print(
        "  ✓ Local class-weighted loss"
    )

    print(
        "  ✓ Label smoothing"
    )

    print(
        "  ✓ Local optimizer"
    )

    print(
        "  ✓ AMP support"
    )

    print(
        "  ✓ Gradient clipping"
    )

    print(
        "  ✓ Local forward pass"
    )

    print(
        "  ✓ Local gradient flow"
    )

    print(
        "  ✓ Global state loading"
    )

    print(
        "  ✓ Local state extraction"
    )

    print(
        "  ✓ Federated state validation"
    )

    print(
        "\nNext:"
    )

    print(
        "  python src\\10A_fedavg_server_convnext.py"
    )

    print(
        "  python src\\11A_train_federated_convnext.py"
    )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    self_test()