"""
08A_train_centralized_convnext_v2.py
------------------------------------
PHASE 2 - STEP 8A (V2)

CONVNEXT-TINY CENTRALIZED BASELINE TRAINING

Purpose:
    Train ConvNeXt-Tiny on the complete centralized training set
    before introducing Federated Learning.

This is an additional centralized reference baseline.

MobileNetV3-Small baseline:
    07_model.py
    08_train_centralized.py

ConvNeXt-Tiny baseline:
    07A_model_convnext.py
    08A_train_centralized_convnext_v2.py

Pipeline:
    Image
       |
       v
    ConvNeXt-Tiny
       |
       v
    768-D deep embedding
       |
       +------------------+
                          |
    1320-D handcrafted --+
       |
       v
    256-D handcrafted branch
       |
       v
    Feature fusion
       |
       v
    1024-D fused representation
       |
       v
    7-class classifier

V2 IMPROVEMENTS:
    1. ConvNeXt-Tiny pretrained on ImageNet
    2. Full backbone fine-tuning
    3. Weighted CrossEntropy for class imbalance
    4. Label smoothing
    5. AdamW optimizer
    6. ReduceLROnPlateau scheduler
    7. Gradient accumulation
    8. AMP mixed precision
    9. Gradient clipping
   10. Early stopping
   11. Best validation Macro-F1 checkpoint
   12. Detailed test evaluation
   13. Per-class F1
   14. Confusion matrix
   15. Training curves
   16. Learning-rate tracking
   17. GPU memory monitoring

IMPORTANT:
    This script uses the existing:
        06_dataset.py
        07A_model_convnext.py

    Do NOT modify the MobileNet files.

    Step 6 already handles:
        - train-only StandardScaler
        - handcrafted feature caching
        - RGB conversion
        - training augmentation
        - ImageNet normalization
        - 224x224 images
        - 1320-D handcrafted features

GPU:
    Designed to work with an RTX 3050 Laptop GPU with 4 GB VRAM.
"""


# =====================================================================
# IMPORTS
# =====================================================================

import os
import sys
import time
import importlib.util
import random

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
# LOAD PYTHON MODULE
# =====================================================================

def _load_module(filename, module_name):

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
            f"Could not load module:\n{path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


# =====================================================================
# LOAD DATASET AND MODEL
# =====================================================================

dataset_module = _load_module(
    "06_dataset.py",
    "dataset_module_convnext_v2"
)

model_module = _load_module(
    "07A_model_convnext.py",
    "model_module_convnext_v2"
)


build_centralized_train_loader = (
    dataset_module.build_centralized_train_loader
)

build_eval_loader = (
    dataset_module.build_eval_loader
)

HybridClassifier = (
    model_module.HybridClassifier
)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_random_seeds():

    seed = config.RANDOM_SEED

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # ---------------------------------------------------------------
    # Reproducibility settings
    # ---------------------------------------------------------------

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(
        f"Random seed: {seed}"
    )


# =====================================================================
# CONFIGURATION HELPERS
# =====================================================================

def get_config_value(name, default):

    return getattr(
        config,
        name,
        default
    )


# =====================================================================
# CLASS WEIGHTS
# =====================================================================

def compute_class_weights(
    device
):

    train_csv = os.path.join(
        config.OUTPUT_DIR,
        "train_split.csv"
    )

    if not os.path.exists(train_csv):

        raise FileNotFoundError(
            f"\nTraining split not found:\n"
            f"{train_csv}"
        )

    train_df = pd.read_csv(
        train_csv
    )

    counts = train_df["label"].value_counts()

    total = len(train_df)

    num_classes = config.NUM_CLASSES

    weights = []

    print(
        "\nTraining class counts:"
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
    # Inverse-frequency class weighting
    # ---------------------------------------------------------------

    for cls in config.CLASS_LIST:

        count = counts.get(
            cls,
            0
        )

        if count == 0:

            weight = 0.0

        else:

            weight = (
                total /
                (
                    num_classes *
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
        "\nClass weights:"
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
# EVALUATION
# =====================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_preds = []

    all_labels = []

    # ---------------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------------

    for images, features, labels in loader:

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
            loss.item() *
            batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_preds.extend(
            predictions.cpu().numpy().tolist()
        )

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

    if len(loader.dataset) == 0:

        raise ValueError(
            "Evaluation DataLoader contains zero samples."
        )

    average_loss = (
        total_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        all_preds,
        all_labels
    )


# =====================================================================
# TRAIN ONE EPOCH
# =====================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    use_amp,
    accumulation_steps,
    grad_clip
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------

    progress = tqdm(
        loader,
        desc="  train",
        leave=False
    )

    for batch_idx, (
        images,
        features,
        labels
    ) in enumerate(progress):

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

        # -----------------------------------------------------------
        # Forward pass
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

                scaled_loss = (
                    loss /
                    accumulation_steps
                )

            scaler.scale(
                scaled_loss
            ).backward()

        else:

            logits = model(
                images,
                features
            )

            loss = criterion(
                logits,
                labels
            )

            scaled_loss = (
                loss /
                accumulation_steps
            )

            scaled_loss.backward()

        # -----------------------------------------------------------
        # Statistics
        # -----------------------------------------------------------

        batch_size = images.size(0)

        total_loss += (
            loss.item() *
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

        # -----------------------------------------------------------
        # Optimizer step
        # -----------------------------------------------------------

        is_last_batch = (
            batch_idx ==
            len(loader) - 1
        )

        should_step = (
            ((batch_idx + 1) % accumulation_steps == 0)
            or
            is_last_batch
        )

        if should_step:

            if use_amp:

                scaler.unscale_(
                    optimizer
                )

            # -------------------------------------------------------
            # Gradient clipping
            # -------------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )

            if use_amp:

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

    if total == 0:

        raise ValueError(
            "Training DataLoader contains zero samples."
        )

    average_loss = (
        total_loss /
        total
    )

    accuracy = (
        correct /
        total
    )

    return (
        average_loss,
        accuracy
    )


# =====================================================================
# TRAINING CURVES
# =====================================================================

def plot_training_curves(
    log_df,
    save_path
):

    if len(log_df) == 0:

        print(
            "No training history available for plotting."
        )

        return

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 5)
    )

    # ---------------------------------------------------------------
    # Loss
    # ---------------------------------------------------------------

    axes[0].plot(
        log_df["epoch"],
        log_df["train_loss"],
        label="Train Loss",
        marker="o"
    )

    axes[0].plot(
        log_df["epoch"],
        log_df["val_loss"],
        label="Validation Loss",
        marker="o"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].set_title(
        "ConvNeXt-Tiny V2 - Loss"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    # ---------------------------------------------------------------
    # Accuracy / Macro-F1
    # ---------------------------------------------------------------

    axes[1].plot(
        log_df["epoch"],
        log_df["train_acc"],
        label="Train Accuracy",
        marker="o"
    )

    axes[1].plot(
        log_df["epoch"],
        log_df["val_acc"],
        label="Validation Accuracy",
        marker="o"
    )

    axes[1].plot(
        log_df["epoch"],
        log_df["val_macro_f1"],
        label="Validation Macro-F1",
        marker="s"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Score"
    )

    axes[1].set_title(
        "ConvNeXt-Tiny V2 - Metrics"
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.3
    )

    # ---------------------------------------------------------------
    # Learning rate
    # ---------------------------------------------------------------

    axes[2].plot(
        log_df["epoch"],
        log_df["learning_rate"],
        label="Learning Rate",
        marker="o"
    )

    axes[2].set_xlabel(
        "Epoch"
    )

    axes[2].set_ylabel(
        "Learning Rate"
    )

    axes[2].set_title(
        "ConvNeXt-Tiny V2 - LR Schedule"
    )

    axes[2].set_yscale(
        "log"
    )

    axes[2].legend()

    axes[2].grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved training curves:\n"
        f"  {save_path}"
    )


# =====================================================================
# CONFUSION MATRIX
# =====================================================================

def plot_confusion_matrix(
    all_labels,
    all_preds,
    class_list,
    save_path
):

    class_indices = list(
        range(
            len(class_list)
        )
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=class_indices
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        cm
    )

    ax.set_xticks(
        class_indices
    )

    ax.set_yticks(
        class_indices
    )

    ax.set_xticklabels(
        class_list
    )

    ax.set_yticklabels(
        class_list
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_title(
        "ConvNeXt-Tiny V2 - Test Confusion Matrix"
    )

    threshold = (
        cm.max() / 2
        if cm.size > 0
        else 0
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

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
                color=text_color,
                fontsize=9
            )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved confusion matrix:\n"
        f"  {save_path}"
    )


# =====================================================================
# GPU MEMORY REPORT
# =====================================================================

def print_gpu_memory(
    title="GPU memory"
):

    if not torch.cuda.is_available():

        return

    allocated = (
        torch.cuda.memory_allocated()
        /
        (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved()
        /
        (1024 ** 2)
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        /
        (1024 ** 2)
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        /
        (1024 ** 2)
    )

    print(
        f"\n{title}:"
    )

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


# =====================================================================
# MAIN
# =====================================================================

def main():

    start_time = time.perf_counter()

    print(
        "=" * 60
    )

    print(
        "PHASE 2 - STEP 8A: "
        "CONVNEXT-TINY CENTRALIZED TRAINING V2"
    )

    print(
        "=" * 60
    )

    # =================================================================
    # DEVICE
    # =================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
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

        torch.cuda.reset_peak_memory_stats()

    # =================================================================
    # RANDOM SEEDS
    # =================================================================

    set_random_seeds()

    # =================================================================
    # TRAINING CONFIGURATION
    # =================================================================

    use_amp = (
        get_config_value(
            "USE_AMP",
            True
        )
        and
        torch.cuda.is_available()
    )

    # ---------------------------------------------------------------
    # ConvNeXt-specific batch size
    #
    # RTX 3050 4 GB:
    # batch_size=8 + accumulation=2
    # gives effective batch size 16.
    # ---------------------------------------------------------------

    batch_size = get_config_value(
        "CONVNEXT_BATCH_SIZE",
        8
    )

    accumulation_steps = get_config_value(
        "CONVNEXT_GRADIENT_ACCUMULATION_STEPS",
        2
    )

    num_epochs = get_config_value(
        "CONVNEXT_EPOCHS",
        50
    )

    patience = get_config_value(
        "CONVNEXT_EARLY_STOPPING_PATIENCE",
        8
    )

    learning_rate = get_config_value(
        "CONVNEXT_LEARNING_RATE",
        config.LEARNING_RATE
    )

    weight_decay = get_config_value(
        "CONVNEXT_WEIGHT_DECAY",
        config.WEIGHT_DECAY
    )

    label_smoothing = get_config_value(
        "CONVNEXT_LABEL_SMOOTHING",
        0.05
    )

    lr_scheduler_factor = get_config_value(
        "CONVNEXT_LR_SCHEDULER_FACTOR",
        0.5
    )

    lr_scheduler_patience = get_config_value(
        "CONVNEXT_LR_SCHEDULER_PATIENCE",
        3
    )

    lr_scheduler_min_lr = get_config_value(
        "CONVNEXT_LR_SCHEDULER_MIN_LR",
        1e-6
    )

    grad_clip = get_config_value(
        "CONVNEXT_GRAD_CLIP",
        1.0
    )

    # =================================================================
    # PRINT CONFIGURATION
    # =================================================================

    print(
        "\nTraining configuration:"
    )

    print(
        f"  Architecture          : ConvNeXt-Tiny"
    )

    print(
        f"  Maximum epochs        : "
        f"{num_epochs}"
    )

    print(
        f"  Batch size             : "
        f"{batch_size}"
    )

    print(
        f"  Gradient accumulation  : "
        f"{accumulation_steps}"
    )

    print(
        f"  Effective batch size   : "
        f"{batch_size * accumulation_steps}"
    )

    print(
        f"  Early stopping         : "
        f"{patience} epochs"
    )

    print(
        f"  Learning rate          : "
        f"{learning_rate}"
    )

    print(
        f"  Weight decay           : "
        f"{weight_decay}"
    )

    print(
        f"  Label smoothing        : "
        f"{label_smoothing}"
    )

    print(
        f"  LR scheduler factor    : "
        f"{lr_scheduler_factor}"
    )

    print(
        f"  LR scheduler patience  : "
        f"{lr_scheduler_patience}"
    )

    print(
        f"  LR scheduler min LR    : "
        f"{lr_scheduler_min_lr}"
    )

    print(
        f"  AMP                    : "
        f"{use_amp}"
    )

    print(
        f"  Gradient clipping      : "
        f"{grad_clip}"
    )

    print(
        f"  Classes                : "
        f"{config.NUM_CLASSES}"
    )

    print(
        f"  Handcrafted features   : "
        f"{config.HANDCRAFTED_FEATURE_DIM}"
    )

    print(
        f"  Image size             : "
        f"{config.IMAGE_SIZE}x{config.IMAGE_SIZE}"
    )

    # =================================================================
    # BUILD DATALOADERS
    # =================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "BUILDING DATALOADERS"
    )

    print(
        "=" * 60
    )

    train_loader = (
        build_centralized_train_loader(
            train_features=None
        )
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # The normal Step 6 loader uses config.BATCH_SIZE.
    #
    # For ConvNeXt V2 we explicitly rebuild the centralized loader
    # with a smaller batch size if necessary.
    # ---------------------------------------------------------------

    if train_loader.batch_size != batch_size:

        train_loader = (
            dataset_module.create_loader(
                dataframe=train_loader.dataset.df,
                feature_lookup=train_loader.dataset.feature_lookup,
                shuffle=True,
                batch_size=batch_size,
                training=True
            )
        )

    val_loader = build_eval_loader(
        "val"
    )

    print(
        f"\nTrain samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Train batch size: "
        f"{train_loader.batch_size}"
    )

    # =================================================================
    # BUILD MODEL
    # =================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "BUILDING CONVNEXT-TINY HYBRID MODEL"
    )

    print(
        "=" * 60
    )

    model = HybridClassifier(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=(
            config.HANDCRAFTED_FEATURE_DIM
        ),
        pretrained=True
    ).to(device)

    # ---------------------------------------------------------------
    # Channels-last memory layout
    # ---------------------------------------------------------------

    if torch.cuda.is_available():

        model = model.to(
            memory_format=torch.channels_last
        )

    print(
        f"\nModel device: "
        f"{device}"
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"Trainable percentage: "
        f"{100.0 * trainable_params / total_params:.2f}%"
    )

    # =================================================================
    # LOSS
    # =================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "LOSS FUNCTION"
    )

    print(
        "=" * 60
    )

    class_weights = compute_class_weights(
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )

    print(
        f"\nUsing weighted CrossEntropyLoss"
    )

    print(
        f"Label smoothing: "
        f"{label_smoothing}"
    )

    # =================================================================
    # OPTIMIZER
    # =================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "OPTIMIZER"
    )

    print(
        "=" * 60
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    print(
        f"\nOptimizer: AdamW"
    )

    print(
        f"Learning rate: "
        f"{learning_rate}"
    )

    print(
        f"Weight decay: "
        f"{weight_decay}"
    )

    # =================================================================
    # LR SCHEDULER
    # =================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "LEARNING RATE SCHEDULER"
    )

    print(
        "=" * 60
    )

    lr_scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=lr_scheduler_min_lr
        )
    )

    print(
        "\nScheduler: ReduceLROnPlateau"
    )

    print(
        "Monitoring: Validation Macro-F1"
    )

    print(
        f"Factor: {lr_scheduler_factor}"
    )

    print(
        f"Patience: {lr_scheduler_patience}"
    )

    print(
        f"Minimum LR: {lr_scheduler_min_lr}"
    )

    # =================================================================
    # AMP
    # =================================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    # =================================================================
    # OUTPUT PATHS
    # =================================================================

    model_save_path = os.path.join(
        config.MODEL_SAVE_DIR,
        "centralized_convnext_baseline_v2.pt"
    )

    log_path = os.path.join(
        config.TRAINING_LOG_DIR,
        "centralized_convnext_training_log_v2.csv"
    )

    plots_dir = os.path.join(
        config.OUTPUT_DIR,
        "plots"
    )

    os.makedirs(
        plots_dir,
        exist_ok=True
    )

    # =================================================================
    # TRAINING STATE
    # =================================================================

    best_val_f1 = -1.0

    best_val_loss = float(
        "inf"
    )

    best_val_acc = 0.0

    best_epoch = 0

    epochs_without_improvement = 0

    log_rows = []

    # =================================================================
    # START TRAINING
    # =================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "STARTING CONVNEXT-TINY CENTRALIZED TRAINING V2"
    )

    print(
        "=" * 60
    )

    for epoch in range(
        1,
        num_epochs + 1
    ):

        epoch_start = time.perf_counter()

        # -------------------------------------------------------------
        # TRAIN
        # -------------------------------------------------------------

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            accumulation_steps=accumulation_steps,
            grad_clip=grad_clip
        )

        # -------------------------------------------------------------
        # VALIDATION
        # -------------------------------------------------------------

        (
            val_loss,
            val_acc,
            val_macro_f1,
            _,
            _
        ) = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion
        )

        # -------------------------------------------------------------
        # Scheduler
        # -------------------------------------------------------------

        current_lr_before = (
            optimizer.param_groups[0]["lr"]
        )

        lr_scheduler.step(
            val_macro_f1
        )

        current_lr_after = (
            optimizer.param_groups[0]["lr"]
        )

        # -------------------------------------------------------------
        # Epoch timing
        # -------------------------------------------------------------

        epoch_time = (
            time.perf_counter()
            -
            epoch_start
        )

        # -------------------------------------------------------------
        # Print epoch results
        # -------------------------------------------------------------

        print(
            f"\nEpoch "
            f"{epoch:02d}/{num_epochs}"
        )

        print(
            f"  Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"  Train Accuracy  : "
            f"{train_acc:.4f}"
        )

        print(
            f"  Val Loss        : "
            f"{val_loss:.4f}"
        )

        print(
            f"  Val Accuracy    : "
            f"{val_acc:.4f}"
        )

        print(
            f"  Val Macro-F1    : "
            f"{val_macro_f1:.4f}"
        )

        print(
            f"  Learning Rate   : "
            f"{current_lr_after:.2e}"
        )

        if (
            current_lr_after
            <
            current_lr_before
        ):

            print(
                f"  -> LR reduced: "
                f"{current_lr_before:.2e}"
                f" -> "
                f"{current_lr_after:.2e}"
            )

        print(
            f"  Epoch time      : "
            f"{epoch_time:.1f}s"
        )

        # -------------------------------------------------------------
        # Log
        # -------------------------------------------------------------

        log_rows.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "train_acc": train_acc,

            "val_loss": val_loss,

            "val_acc": val_acc,

            "val_macro_f1": val_macro_f1,

            "learning_rate": current_lr_after,

            "epoch_time_sec": epoch_time,

        })

        # -------------------------------------------------------------
        # Checkpoint
        # -------------------------------------------------------------

        if val_macro_f1 > best_val_f1:

            best_val_f1 = (
                val_macro_f1
            )

            best_val_loss = (
                val_loss
            )

            best_val_acc = (
                val_acc
            )

            best_epoch = (
                epoch
            )

            epochs_without_improvement = 0

            # ---------------------------------------------------------
            # Save checkpoint
            # ---------------------------------------------------------

            torch.save({

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    lr_scheduler.state_dict(),

                "epoch":
                    epoch,

                "val_macro_f1":
                    val_macro_f1,

                "val_loss":
                    val_loss,

                "val_acc":
                    val_acc,

                "class_list":
                    config.CLASS_LIST,

                "num_classes":
                    config.NUM_CLASSES,

                "handcrafted_feature_dim":
                    config.HANDCRAFTED_FEATURE_DIM,

                "random_seed":
                    config.RANDOM_SEED,

                "architecture":
                    "ConvNeXt-Tiny",

                "deep_embedding_dim":
                    768,

                "fusion_dimension":
                    1024,

                "learning_rate":
                    learning_rate,

                "weight_decay":
                    weight_decay,

                "label_smoothing":
                    label_smoothing,

                "batch_size":
                    batch_size,

                "gradient_accumulation_steps":
                    accumulation_steps,

            }, model_save_path)

            print(
                f"\n  ✓ New best validation "
                f"Macro-F1: "
                f"{val_macro_f1:.4f}"
            )

            print(
                "  ✓ Best checkpoint saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\n  No improvement. "
                f"({epochs_without_improvement}/"
                f"{patience})"
            )

            # ---------------------------------------------------------
            # Early stopping
            # ---------------------------------------------------------

            if (
                epochs_without_improvement
                >=
                patience
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

        # -------------------------------------------------------------
        # GPU memory
        # -------------------------------------------------------------

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                /
                (1024 ** 2)
            )

            print(
                f"  GPU allocated  : "
                f"{allocated:.2f} MB"
            )

    # =================================================================
    # SAVE TRAINING LOG
    # =================================================================

    log_df = pd.DataFrame(
        log_rows
    )

    log_df.to_csv(
        log_path,
        index=False
    )

    print(
        f"\nTraining log saved:\n"
        f"  {log_path}"
    )

    # =================================================================
    # PLOT TRAINING CURVES
    # =================================================================

    curve_path = os.path.join(
        plots_dir,
        "centralized_convnext_training_curve_v2.png"
    )

    plot_training_curves(
        log_df,
        curve_path
    )

    # =================================================================
    # LOAD BEST CHECKPOINT
    # =================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "LOADING BEST CONVNEXT CHECKPOINT"
    )

    print(
        "=" * 60
    )

    if not os.path.exists(
        model_save_path
    ):

        raise FileNotFoundError(
            f"\nBest checkpoint was not created:\n"
            f"{model_save_path}"
        )

    checkpoint = torch.load(
        model_save_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
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

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 60
    )

    test_loader = build_eval_loader(
        "test"
    )

    (
        test_loss,
        test_acc,
        test_macro_f1,
        test_preds,
        test_labels
    ) = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion
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
            range(
                config.NUM_CLASSES
            )
        ),
        target_names=config.CLASS_LIST,
        zero_division=0
    )

    print(
        report
    )

    # =================================================================
    # SAVE CLASSIFICATION REPORT
    # =================================================================

    report_path = os.path.join(
        config.TRAINING_LOG_DIR,
        "centralized_convnext_test_classification_report_v2.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CONVNEXT-TINY CENTRALIZED "
            "BASELINE V2 TEST CLASSIFICATION REPORT\n\n"
        )

        f.write(
            report
        )

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
            f"\nBest Validation Macro-F1: "
            f"{best_val_f1:.6f}\n"
        )

        f.write(
            f"Best Validation Accuracy: "
            f"{best_val_acc:.6f}\n"
        )

        f.write(
            f"Best Epoch: "
            f"{best_epoch}\n"
        )

    print(
        f"Saved classification report:\n"
        f"  {report_path}"
    )

    # =================================================================
    # PER-CLASS F1
    # =================================================================

    print(
        "\nTest F1 by class:"
    )

    per_class_f1 = f1_score(
        test_labels,
        test_preds,
        labels=list(
            range(
                config.NUM_CLASSES
            )
        ),
        average=None,
        zero_division=0
    )

    for cls, score in zip(
        config.CLASS_LIST,
        per_class_f1
    ):

        print(
            f"  {cls:6s}: "
            f"{score:.4f}"
        )

    # =================================================================
    # SAVE PER-CLASS F1
    # =================================================================

    per_class_f1_path = os.path.join(
        config.TRAINING_LOG_DIR,
        "centralized_convnext_per_class_f1_v2.csv"
    )

    per_class_df = pd.DataFrame({

        "class":
            config.CLASS_LIST,

        "f1_score":
            per_class_f1

    })

    per_class_df.to_csv(
        per_class_f1_path,
        index=False
    )

    print(
        f"\nSaved per-class F1:\n"
        f"  {per_class_f1_path}"
    )

    # =================================================================
    # CONFUSION MATRIX
    # =================================================================

    confusion_path = os.path.join(
        plots_dir,
        "centralized_convnext_confusion_matrix_v2.png"
    )

    plot_confusion_matrix(
        test_labels,
        test_preds,
        config.CLASS_LIST,
        confusion_path
    )

    # =================================================================
    # GPU MEMORY
    # =================================================================

    print_gpu_memory(
        "GPU memory"
    )

    # =================================================================
    # TOTAL TIME
    # =================================================================

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 2 - STEP 8A COMPLETE ✓"
    )

    print(
        "=" * 60
    )

    print(
        f"Total time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        "\n" + "-" * 60
    )

    print(
        "CONVNEXT-TINY CENTRALIZED REFERENCE BASELINE V2"
    )

    print(
        "-" * 60
    )

    print(
        f"Best Val Macro-F1 : "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Best Val Accuracy : "
        f"{best_val_acc:.4f}"
    )

    print(
        f"Best Epoch        : "
        f"{best_epoch}"
    )

    print(
        f"Test Accuracy     : "
        f"{test_acc:.4f}"
    )

    print(
        f"Test Macro-F1     : "
        f"{test_macro_f1:.4f}"
    )

    print(
        "-" * 60
    )

    # =================================================================
    # OUTPUTS
    # =================================================================

    print(
        "\nSaved outputs:"
    )

    print(
        f"  Model:\n"
        f"    {model_save_path}"
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
        f"    {per_class_f1_path}"
    )

    print(
        f"  Training curves:\n"
        f"    {curve_path}"
    )

    print(
        f"  Confusion matrix:\n"
        f"    {confusion_path}"
    )

    # =================================================================
    # COMPARISON
    # =================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "CENTRALIZED MODEL COMPARISON"
    )

    print(
        "=" * 60
    )

    print(
        "\nMobileNetV3-Small:"
    )

    print(
        "  Model: centralized_baseline.pt"
    )

    print(
        "  See its saved test results."
    )

    print(
        "\nConvNeXt-Tiny V2:"
    )

    print(
        f"  Test Accuracy = "
        f"{test_acc:.4f}"
    )

    print(
        f"  Test Macro-F1 = "
        f"{test_macro_f1:.4f}"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "NEXT PHASE"
    )

    print(
        "=" * 60
    )

    print(
        "\nThe centralized ConvNeXt-Tiny V2 baseline "
        "is complete."
    )

    print(
        "\nNext:"
    )

    print(
        "  python src\\09_federated_client.py"
    )

    print(
        "  python src\\10_fedavg_server.py"
    )

    print(
        "  python src\\11_train_federated.py"
    )

    print(
        "\nThe ConvNeXt-Tiny centralized result should "
        "be used as a reference when evaluating the "
        "federated ConvNeXt experiments."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()