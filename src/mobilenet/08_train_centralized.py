"""
08_train_centralized.py
-----------------------
PHASE 2 - STEP 8: CENTRALIZED BASELINE TRAINING

Purpose:
    Train the HybridClassifier using the complete pooled training set
    before introducing Federated Learning.

Pipeline:
    Image
        +
    1320-D handcrafted features
        |
        v
    HybridClassifier
        |
        v
    7-class prediction

Important:
    - Training augmentation is handled by 06_dataset.py.
    - Validation/test images receive NO augmentation.
    - StandardScaler is fitted ONLY on training handcrafted features.
    - Validation Macro-F1 is the model-selection metric.
    - Test data is NEVER used for checkpoint selection.
    - Best validation checkpoint is restored before final testing.

Designed for:
    NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)
"""


# =====================================================================
# IMPORTS
# =====================================================================

import os
import sys
import time
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
# LOAD DATASET / MODEL MODULES
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
    "07_model.py",
    "model_module"
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
    # Deterministic behavior
    #
    # NOTE:
    # Deterministic mode can be slightly slower.
    # It is useful for an academic/research experiment.
    # ---------------------------------------------------------------

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

    total = len(
        train_df
    )

    num_classes = config.NUM_CLASSES

    weights = []

    print(
        "\nTraining class counts:"
    )

    for class_name in config.CLASS_LIST:

        count = int(
            counts.get(
                class_name,
                0
            )
        )

        print(
            f"  {class_name:6s}: {count}"
        )

    # ---------------------------------------------------------------
    # Standard balanced-class weighting
    #
    # weight_c = N / (C * n_c)
    #
    # This compensates for the strong HAM10000 class imbalance.
    # ---------------------------------------------------------------

    for class_name in config.CLASS_LIST:

        count = int(
            counts.get(
                class_name,
                0
            )
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

    for class_name, weight in zip(
        config.CLASS_LIST,
        weights
    ):

        print(
            f"  {class_name:6s}: {weight:.4f}"
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
    use_amp
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    progress_bar = tqdm(
        loader,
        desc="  train",
        leave=False
    )

    for images, features, labels in progress_bar:

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

            # -------------------------------------------------------
            # Unscale before gradient clipping
            # -------------------------------------------------------

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
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
                max_norm=1.0
            )

            optimizer.step()

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

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
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
        "Centralized Baseline - Loss"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    # ---------------------------------------------------------------
    # Accuracy + Macro-F1
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
        "Centralized Baseline - Metrics"
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
        "Centralized Baseline - LR Schedule"
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
        range(len(class_list))
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
        "Centralized Baseline - Test Confusion Matrix"
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
# PER-CLASS F1
# =====================================================================

def calculate_per_class_f1(
    labels,
    predictions,
    class_list
):

    scores = f1_score(
        labels,
        predictions,
        labels=list(
            range(len(class_list))
        ),
        average=None,
        zero_division=0
    )

    print(
        "\nTest F1 by class:"
    )

    for class_name, score in zip(
        class_list,
        scores
    ):

        print(
            f"  {class_name:6s}: {score:.4f}"
        )

    return scores


# =====================================================================
# SAVE TEST RESULTS
# =====================================================================

def save_test_report(
    test_labels,
    test_preds,
    test_loss,
    test_acc,
    test_macro_f1,
    class_list,
    save_path
):

    report = classification_report(
        test_labels,
        test_preds,
        labels=list(
            range(len(class_list))
        ),
        target_names=class_list,
        zero_division=0
    )

    print(
        "\nClassification Report:"
    )

    print(
        report
    )

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CENTRALIZED BASELINE "
            "TEST CLASSIFICATION REPORT\n"
        )

        f.write(
            "=" * 60 +
            "\n\n"
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

    print(
        f"Saved classification report:\n"
        f"  {save_path}"
    )

    return report


# =====================================================================
# MAIN
# =====================================================================

def main():

    start_time = time.perf_counter()

    print(
        "=" * 60
    )

    print(
        "PHASE 2 - STEP 8: "
        "CENTRALIZED BASELINE TRAINING"
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
    # CONFIGURATION
    # =================================================================

    num_epochs = getattr(
        config,
        "NUM_CENTRALIZED_EPOCHS",
        40
    )

    patience = getattr(
        config,
        "EARLY_STOPPING_PATIENCE",
        8
    )

    label_smoothing = getattr(
        config,
        "LABEL_SMOOTHING",
        0.05
    )

    lr_scheduler_factor = getattr(
        config,
        "LR_SCHEDULER_FACTOR",
        0.5
    )

    lr_scheduler_patience = getattr(
        config,
        "LR_SCHEDULER_PATIENCE",
        2
    )

    lr_scheduler_min_lr = getattr(
        config,
        "LR_SCHEDULER_MIN_LR",
        1e-6
    )

    use_amp = (
        getattr(
            config,
            "USE_AMP",
            True
        )
        and
        torch.cuda.is_available()
    )

    grad_clip = 1.0

    print(
        "\nTraining configuration:"
    )

    print(
        f"  Maximum epochs        : "
        f"{num_epochs}"
    )

    print(
        f"  Early stopping        : "
        f"{patience} epochs"
    )

    print(
        f"  Learning rate         : "
        f"{config.LEARNING_RATE}"
    )

    print(
        f"  Weight decay          : "
        f"{config.WEIGHT_DECAY}"
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
        f"  LR minimum            : "
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
        f"  Batch size            : "
        f"{config.BATCH_SIZE}"
    )

    print(
        f"  Image size            : "
        f"{config.IMAGE_SIZE}"
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
        build_centralized_train_loader()
    )

    val_loader = (
        build_eval_loader(
            "val"
        )
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
    # VERIFY FIRST BATCH
    # =================================================================

    train_batch = next(
        iter(train_loader)
    )

    print(
        "\nTraining batch:"
    )

    print(
        f"  Images   : "
        f"{train_batch[0].shape}"
    )

    print(
        f"  Features : "
        f"{train_batch[1].shape}"
    )

    print(
        f"  Labels   : "
        f"{train_batch[2].shape}"
    )

    # =================================================================
    # BUILD MODEL
    # =================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "BUILDING HYBRID MODEL"
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

    class_weights = (
        compute_class_weights(
            device
        )
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )

    print(
        "\nUsing:"
    )

    print(
        "  Weighted CrossEntropyLoss"
    )

    print(
        f"  Label smoothing = "
        f"{label_smoothing}"
    )

    # =================================================================
    # OPTIMIZER
    # =================================================================
    #
    # AdamW is used instead of Adam because decoupled weight decay
    # gives cleaner regularization.
    # =================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    print(
        "\nOptimizer:"
    )

    print(
        "  AdamW"
    )

    print(
        f"  LR = "
        f"{config.LEARNING_RATE}"
    )

    print(
        f"  Weight decay = "
        f"{config.WEIGHT_DECAY}"
    )

    # =================================================================
    # LEARNING RATE SCHEDULER
    # =================================================================
    #
    # Validation Macro-F1 is the optimization signal because the
    # dataset is highly imbalanced.
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
    # AMP
    # =================================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    # =================================================================
    # OUTPUT PATHS
    # =================================================================

    os.makedirs(
        config.MODEL_SAVE_DIR,
        exist_ok=True
    )

    os.makedirs(
        config.TRAINING_LOG_DIR,
        exist_ok=True
    )

    plots_dir = os.path.join(
        config.OUTPUT_DIR,
        "plots"
    )

    os.makedirs(
        plots_dir,
        exist_ok=True
    )

    model_save_path = os.path.join(
        config.MODEL_SAVE_DIR,
        "centralized_baseline.pt"
    )

    log_path = os.path.join(
        config.TRAINING_LOG_DIR,
        "centralized_training_log.csv"
    )

    report_path = os.path.join(
        config.TRAINING_LOG_DIR,
        "centralized_test_classification_report.txt"
    )

    curve_path = os.path.join(
        plots_dir,
        "centralized_training_curve.png"
    )

    confusion_path = os.path.join(
        plots_dir,
        "centralized_confusion_matrix.png"
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
        "STARTING CENTRALIZED TRAINING"
    )

    print(
        "=" * 60
    )

    for epoch in range(
        1,
        num_epochs + 1
    ):

        epoch_start = (
            time.perf_counter()
        )

        # -------------------------------------------------------------
        # TRAIN
        # -------------------------------------------------------------

        train_loss, train_acc = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                use_amp=use_amp
            )
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
        # LR scheduler
        # -------------------------------------------------------------

        lr_before = (
            optimizer
            .param_groups[0]["lr"]
        )

        lr_scheduler.step(
            val_macro_f1
        )

        lr_after = (
            optimizer
            .param_groups[0]["lr"]
        )

        epoch_time = (
            time.perf_counter()
            -
            epoch_start
        )

        # -------------------------------------------------------------
        # PRINT EPOCH
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
            f"{lr_after:.2e}"
        )

        if lr_after < lr_before:

            print(
                f"  -> LR reduced: "
                f"{lr_before:.2e} "
                f"-> "
                f"{lr_after:.2e}"
            )

        print(
            f"  Epoch time      : "
            f"{epoch_time:.1f}s"
        )

        # -------------------------------------------------------------
        # LOG
        # -------------------------------------------------------------

        log_rows.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "train_acc": train_acc,

            "val_loss": val_loss,

            "val_acc": val_acc,

            "val_macro_f1": val_macro_f1,

            "learning_rate": lr_after,

            "epoch_time_sec": epoch_time,

        })

        # -------------------------------------------------------------
        # BEST CHECKPOINT
        #
        # PRIMARY METRIC = VALIDATION MACRO-F1
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

            checkpoint = {

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

                "learning_rate":
                    config.LEARNING_RATE,

                "weight_decay":
                    config.WEIGHT_DECAY,

                "label_smoothing":
                    label_smoothing,

            }

            torch.save(
                checkpoint,
                model_save_path
            )

            print(
                f"\n  ✓ New best "
                f"validation Macro-F1: "
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
            # EARLY STOPPING
            # ---------------------------------------------------------

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
        "LOADING BEST CHECKPOINT"
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

    test_loader = (
        build_eval_loader(
            "test"
        )
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

    report = save_test_report(
        test_labels=test_labels,
        test_preds=test_preds,
        test_loss=test_loss,
        test_acc=test_acc,
        test_macro_f1=test_macro_f1,
        class_list=config.CLASS_LIST,
        save_path=report_path
    )

    # =================================================================
    # PER-CLASS F1
    # =================================================================

    per_class_f1 = (
        calculate_per_class_f1(
            labels=test_labels,
            predictions=test_preds,
            class_list=config.CLASS_LIST
        )
    )

    # =================================================================
    # CONFUSION MATRIX
    # =================================================================

    plot_confusion_matrix(
        all_labels=test_labels,
        all_preds=test_preds,
        class_list=config.CLASS_LIST,
        save_path=confusion_path
    )

    # =================================================================
    # FINAL SUMMARY
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
        "PHASE 2 - STEP 8 COMPLETE ✓"
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
        "CENTRALIZED REFERENCE BASELINE"
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
        f"  Training curves:\n"
        f"    {curve_path}"
    )

    print(
        f"  Confusion matrix:\n"
        f"    {confusion_path}"
    )

    # =================================================================
    # NEXT STEP
    # =================================================================

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
        "\nThe centralized model is now the "
        "reference baseline for the federated experiments."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()