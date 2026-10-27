"""
08A_train_centralized_convnext.py
---------------------------------
PHASE 2 - STEP 8A

CENTRALIZED CONVNEXT-TINY BASELINE TRAINING

Purpose:
    Train the ConvNeXt-Tiny hybrid classifier on ALL pooled training
    data as a centralized reference baseline.

Architecture:
    ConvNeXt-Tiny
        768-D deep embedding
              +
        1320-D handcrafted features
              |
        Handcrafted branch -> 256-D
              |
        Feature fusion -> 1024-D
              |
        Classification head -> 7 classes

Important:
    This file uses the SAME:
        - train/val/test splits
        - handcrafted features
        - StandardScaler
        - image augmentation
        - ImageNet normalization
        - class mapping

as Step 6 and the MobileNetV3-Small centralized baseline.

This makes the ConvNeXt result directly comparable with:
    08_train_centralized.py

GPU:
    Designed for RTX 3050 4 GB VRAM.

Memory optimizations:
    - AMP / mixed precision
    - gradient accumulation
    - batch size taken from config
    - optional CUDA memory reporting

Training:
    - weighted CrossEntropyLoss
    - label smoothing
    - AdamW
    - ReduceLROnPlateau
    - gradient clipping
    - early stopping
    - best checkpoint based on validation Macro-F1

Outputs:
    outputs/models/centralized_convnext_baseline.pt
    outputs/training_logs/centralized_convnext_training_log.csv
    outputs/training_logs/centralized_convnext_test_classification_report.txt
    outputs/plots/centralized_convnext_training_curve.png
    outputs/plots/centralized_convnext_confusion_matrix.png
"""

import os
import sys
import time
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
# LOAD DATASET AND MODEL MODULES
# =====================================================================

def _load_module(filename, module_name):

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

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


dataset_module = _load_module(
    "06_dataset.py",
    "dataset_module"
)

model_module = _load_module(
    "07A_model_convnext.py",
    "model_module_convnext"
)


# Dataset functions
build_centralized_train_loader = (
    dataset_module.build_centralized_train_loader
)

build_eval_loader = (
    dataset_module.build_eval_loader
)


# ConvNeXt model
HybridClassifier = model_module.HybridClassifier


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_random_seeds():

    seed = config.RANDOM_SEED

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        # -------------------------------------------------------------
        # Deterministic settings
        # -------------------------------------------------------------
        #
        # These improve reproducibility.
        #
        # They may reduce performance slightly compared with fully
        # benchmarked CUDA execution.
        # -------------------------------------------------------------

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

    print(
        f"Random seed: {seed}"
    )


# =====================================================================
# CLASS WEIGHTS
# =====================================================================

def compute_class_weights(device):

    train_csv = os.path.join(
        config.OUTPUT_DIR,
        "train_split.csv"
    )

    if not os.path.exists(train_csv):

        raise FileNotFoundError(
            f"Training split not found:\n"
            f"{train_csv}"
        )

    train_df = pd.read_csv(
        train_csv
    )

    counts = train_df[
        "label"
    ].value_counts()

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

    # -------------------------------------------------------------
    # Inverse-frequency balanced weights
    # -------------------------------------------------------------

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
                (num_classes * count)
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
    criterion,
    use_amp
):

    model.eval()

    total_loss = 0.0

    all_preds = []

    all_labels = []

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

        # -------------------------------------------------------------
        # AMP evaluation
        # -------------------------------------------------------------

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

        else:

            logits = model(
                images,
                features
            )

            loss = criterion(
                logits,
                labels
            )

        total_loss += (
            loss.item()
            *
            images.size(0)
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

    # -------------------------------------------------------------
    # Progress bar
    # -------------------------------------------------------------

    progress = tqdm(
        loader,
        desc="  train",
        leave=False
    )

    num_batches = len(loader)

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

        # ---------------------------------------------------------
        # Forward + loss
        # ---------------------------------------------------------

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda",
                enabled=True
            ):

                logits = model(
                    images,
                    features
                )

                raw_loss = criterion(
                    logits,
                    labels
                )

                loss = (
                    raw_loss /
                    accumulation_steps
                )

            scaler.scale(
                loss
            ).backward()

        else:

            logits = model(
                images,
                features
            )

            raw_loss = criterion(
                logits,
                labels
            )

            loss = (
                raw_loss /
                accumulation_steps
            )

            loss.backward()

        # ---------------------------------------------------------
        # Gradient accumulation
        # ---------------------------------------------------------

        should_step = (
            (batch_idx + 1) % accumulation_steps == 0
            or
            (batch_idx + 1) == num_batches
        )

        if should_step:

            if use_amp:

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

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip
                )

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        # ---------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------

        batch_size = images.size(0)

        total_loss += (
            raw_loss.item()
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
            loss=f"{raw_loss.item():.4f}"
        )

    if total == 0:

        raise ValueError(
            "Training DataLoader contains zero samples."
        )

    epoch_loss = (
        total_loss /
        total
    )

    epoch_accuracy = (
        correct /
        total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# =====================================================================
# TRAINING CURVES
# =====================================================================

def plot_training_curves(
    log_df,
    save_path
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 5)
    )

    # -------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------

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
        "ConvNeXt-Tiny - Loss"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    # -------------------------------------------------------------
    # Accuracy / Macro-F1
    # -------------------------------------------------------------

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
        "ConvNeXt-Tiny - Metrics"
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.3
    )

    # -------------------------------------------------------------
    # Learning rate
    # -------------------------------------------------------------

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
        "ConvNeXt-Tiny - LR Schedule"
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
        "ConvNeXt-Tiny - Test Confusion Matrix"
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

def print_gpu_memory():

    if not torch.cuda.is_available():

        return

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

    peak_allocated = (
        torch.cuda.max_memory_allocated(0)
        /
        (1024 ** 2)
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved(0)
        /
        (1024 ** 2)
    )

    print(
        "\nGPU memory:"
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
        "CENTRALIZED CONVNEXT-TINY TRAINING"
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

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            /
            (1024 ** 3)
        )

        print(
            f"GPU VRAM: "
            f"{total_memory:.2f} GB"
        )

    # =================================================================
    # REPRODUCIBILITY
    # =================================================================

    set_random_seeds()

    # =================================================================
    # TRAINING CONFIGURATION
    # =================================================================

    use_amp = (
        getattr(
            config,
            "USE_AMP",
            True
        )
        and
        torch.cuda.is_available()
    )

    # -------------------------------------------------------------
    # ConvNeXt is significantly heavier than MobileNetV3-Small.
    #
    # Keep the physical batch small and use gradient accumulation
    # to obtain a larger effective batch size.
    # -------------------------------------------------------------

    batch_size = getattr(
        config,
        "CONVNEXT_BATCH_SIZE",
        8
    )

    accumulation_steps = getattr(
        config,
        "CONVNEXT_ACCUMULATION_STEPS",
        2
    )

    num_epochs = getattr(
        config,
        "NUM_CONVNEXT_EPOCHS",
        40
    )

    patience = getattr(
        config,
        "CONVNEXT_EARLY_STOPPING_PATIENCE",
        7
    )

    label_smoothing = getattr(
        config,
        "CONVNEXT_LABEL_SMOOTHING",
        0.05
    )

    learning_rate = getattr(
        config,
        "CONVNEXT_LEARNING_RATE",
        3e-4
    )

    weight_decay = getattr(
        config,
        "CONVNEXT_WEIGHT_DECAY",
        1e-4
    )

    lr_scheduler_factor = getattr(
        config,
        "CONVNEXT_LR_SCHEDULER_FACTOR",
        0.5
    )

    lr_scheduler_patience = getattr(
        config,
        "CONVNEXT_LR_SCHEDULER_PATIENCE",
        2
    )

    lr_scheduler_min_lr = getattr(
        config,
        "CONVNEXT_LR_SCHEDULER_MIN_LR",
        1e-6
    )

    grad_clip = getattr(
        config,
        "CONVNEXT_GRAD_CLIP",
        1.0
    )

    effective_batch_size = (
        batch_size *
        accumulation_steps
    )

    print(
        "\nTraining configuration:"
    )

    print(
        f"  Architecture          : ConvNeXt-Tiny"
    )

    print(
        f"  Maximum epochs        : {num_epochs}"
    )

    print(
        f"  Early stopping        : "
        f"{patience} epochs"
    )

    print(
        f"  Physical batch size   : "
        f"{batch_size}"
    )

    print(
        f"  Accumulation steps    : "
        f"{accumulation_steps}"
    )

    print(
        f"  Effective batch size  : "
        f"{effective_batch_size}"
    )

    print(
        f"  Learning rate         : "
        f"{learning_rate}"
    )

    print(
        f"  Weight decay          : "
        f"{weight_decay}"
    )

    print(
        f"  Label smoothing       : "
        f"{label_smoothing}"
    )

    print(
        f"  LR scheduler factor   : "
        f"{lr_scheduler_factor}"
    )

    print(
        f"  LR scheduler patience : "
        f"{lr_scheduler_patience}"
    )

    print(
        f"  Minimum LR            : "
        f"{lr_scheduler_min_lr}"
    )

    print(
        f"  AMP                   : "
        f"{use_amp}"
    )

    print(
        f"  Gradient clipping     : "
        f"{grad_clip}"
    )

    print(
        f"  Classes               : "
        f"{config.NUM_CLASSES}"
    )

    print(
        f"  Handcrafted features  : "
        f"{config.HANDCRAFTED_FEATURE_DIM}"
    )

    # =================================================================
    # DATALOADERS
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

    # -------------------------------------------------------------
    # Important:
    # These loaders use 06_dataset.py.
    #
    # Therefore:
    #   Train -> augmentation
    #   Val   -> no augmentation
    #   Test  -> no augmentation
    #
    # Handcrafted features remain the same scaled 1320-D features.
    # -------------------------------------------------------------

    train_loader = build_centralized_train_loader()

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
        handcrafted_feature_dim=config.HANDCRAFTED_FEATURE_DIM,
        pretrained=True
    ).to(device)

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
    # LOSS FUNCTION
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
        "\nUsing weighted CrossEntropyLoss"
        f" (label_smoothing={label_smoothing})."
    )

    # =================================================================
    # OPTIMIZER
    # =================================================================

    # -------------------------------------------------------------
    # AdamW is generally preferable for this modern CNN backbone.
    # -------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # =================================================================
    # LEARNING RATE SCHEDULER
    # =================================================================

    lr_scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=lr_scheduler_min_lr
        )
    )

    # =================================================================
    # AMP SCALER
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
        "centralized_convnext_baseline.pt"
    )

    log_path = os.path.join(
        config.TRAINING_LOG_DIR,
        "centralized_convnext_training_log.csv"
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

    best_val_accuracy = 0.0

    best_epoch = 0

    epochs_without_improvement = 0

    log_rows = []

    # =================================================================
    # CUDA MEMORY RESET
    # =================================================================

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    # =================================================================
    # START TRAINING
    # =================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "STARTING CONVNEXT-TINY CENTRALIZED TRAINING"
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
            criterion=criterion,
            use_amp=use_amp
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

        if current_lr_after < current_lr_before:

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

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_acc":
                train_acc,

            "val_loss":
                val_loss,

            "val_acc":
                val_acc,

            "val_macro_f1":
                val_macro_f1,

            "learning_rate":
                current_lr_after,

            "epoch_time_sec":
                epoch_time,

        })

        # -------------------------------------------------------------
        # BEST MODEL
        # -------------------------------------------------------------

        if val_macro_f1 > best_val_f1:

            best_val_f1 = val_macro_f1

            best_val_loss = val_loss

            best_val_accuracy = val_acc

            best_epoch = epoch

            epochs_without_improvement = 0

            # ---------------------------------------------------------
            # Save complete checkpoint
            # ---------------------------------------------------------

            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

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

                "handcrafted_branch_dim":
                    256,

                "fusion_dim":
                    1024,

                "learning_rate":
                    learning_rate,

                "weight_decay":
                    weight_decay,

                "label_smoothing":
                    label_smoothing,

                "batch_size":
                    batch_size,

                "accumulation_steps":
                    accumulation_steps,

            }

            torch.save(
                checkpoint,
                model_save_path
            )

            print(
                f"\n  ✓ New best validation "
                f"Macro-F1: {val_macro_f1:.4f}"
            )

            print(
                "  ✓ Best ConvNeXt checkpoint saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\n  No improvement. "
                f"({epochs_without_improvement}/{patience})"
            )

            if (
                epochs_without_improvement
                >= patience
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

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
    # TRAINING CURVES
    # =================================================================

    curve_path = os.path.join(
        plots_dir,
        "centralized_convnext_training_curve.png"
    )

    plot_training_curves(
        log_df,
        curve_path
    )

    # =================================================================
    # GPU MEMORY
    # =================================================================

    print_gpu_memory()

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
            f"Best model checkpoint was not created:\n"
            f"{model_save_path}"
        )

    checkpoint = torch.load(
        model_save_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
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
    # TEST DATASET
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
        criterion=criterion,
        use_amp=use_amp
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

    report_path = os.path.join(
        config.TRAINING_LOG_DIR,
        "centralized_convnext_test_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CONVNEXT-TINY CENTRALIZED BASELINE "
            "TEST CLASSIFICATION REPORT\n\n"
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
            f"\nBest Epoch: "
            f"{checkpoint['epoch']}\n"
        )

        f.write(
            f"Best Validation Macro-F1: "
            f"{checkpoint['val_macro_f1']:.6f}\n"
        )

        f.write(
            f"Best Validation Accuracy: "
            f"{checkpoint['val_acc']:.6f}\n"
        )

    print(
        f"Saved classification report:\n"
        f"  {report_path}"
    )

    # =================================================================
    # PER-CLASS F1
    # =================================================================

    test_f1_scores = f1_score(
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

    print(
        "\nTest F1 by class:"
    )

    for class_name, score in zip(
        config.CLASS_LIST,
        test_f1_scores
    ):

        print(
            f"  {class_name:6s}: "
            f"{score:.4f}"
        )

    # =================================================================
    # CONFUSION MATRIX
    # =================================================================

    confusion_path = os.path.join(
        plots_dir,
        "centralized_convnext_confusion_matrix.png"
    )

    plot_confusion_matrix(
        test_labels,
        test_preds,
        config.CLASS_LIST,
        confusion_path
    )

    # =================================================================
    # FINAL GPU MEMORY
    # =================================================================

    print_gpu_memory()

    # =================================================================
    # SUMMARY
    # =================================================================

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

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
        "CONVNEXT-TINY CENTRALIZED REFERENCE BASELINE"
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
        f"{best_val_accuracy:.4f}"
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
        f"  Training curves:\n"
        f"    {curve_path}"
    )

    print(
        f"  Confusion matrix:\n"
        f"    {confusion_path}"
    )

    print(
        "\nComparison:"
    )

    print(
        "  MobileNetV3-Small:"
    )

    print(
        "    See centralized_baseline.pt"
    )

    print(
        "  ConvNeXt-Tiny:"
    )

    print(
        f"    Test Accuracy = "
        f"{test_acc:.4f}"
    )

    print(
        f"    Test Macro-F1 = "
        f"{test_macro_f1:.4f}"
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
        "\nThe ConvNeXt-Tiny model is now available "
        "as an additional centralized reference "
        "for comparison with MobileNetV3-Small."
    )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    main()