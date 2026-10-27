"""
08G_train_centralized_ensemble.py

Centralized joint training for the ConvNeXt-Tiny + MaxViT-Tiny hybrid ensemble.
Uses the existing 06_dataset.py and 07G_ensemble_model.py without changing
either original backbone implementation.

Expected location: src/08G_train_centralized_ensemble.py
Sibling files:
    06_dataset.py
    07G_ensemble_model.py
    07A_model_convnext.py
    07F_model_maxvit_tiny.py

Outputs are written under config.TRAINING_LOG_DIR and config.OUTPUT_DIR.
"""

from __future__ import annotations

import os
import sys
import time
import random
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE.parent / "configs"
if CONFIG_DIR.exists():
    sys.path.insert(0, str(CONFIG_DIR))
else:
    CONFIG_DIR = HERE / "configs"
    if CONFIG_DIR.exists():
        sys.path.insert(0, str(CONFIG_DIR))

import config


def load_module(filename: str, module_name: str):
    path = HERE / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Place {filename} beside this training script."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


dataset_module = load_module("06_dataset.py", "ensemble_dataset_module")
ensemble_module = load_module("07G_ensemble_model.py", "ensemble_model_module")

build_centralized_train_loader = dataset_module.build_centralized_train_loader
build_eval_loader = dataset_module.build_eval_loader
EnsembleClassifier = ensemble_module.ConvNeXtMaxViTEnsemble


def cfg(name, default):
    return getattr(config, name, default)


def set_seeds():
    seed = int(cfg("RANDOM_SEED", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Repeatability without forcing deterministic algorithms that may be
    # unsupported by some CUDA kernels.
    torch.backends.cudnn.benchmark = False
    print(f"Random seed: {seed}")


def make_dirs():
    output_dir = Path(config.OUTPUT_DIR)
    log_dir = Path(cfg("TRAINING_LOG_DIR", output_dir / "training_logs"))
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, log_dir, plots_dir


def compute_class_weights(device):
    train_csv = Path(config.OUTPUT_DIR) / "train_split.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(
            f"Training split not found: {train_csv}\n"
            "Run the project data preparation step first."
        )
    df = pd.read_csv(train_csv)
    if "label" not in df.columns:
        raise ValueError(f"'label' column missing from {train_csv}")

    counts = df["label"].value_counts()
    total = len(df)
    n_classes = int(config.NUM_CLASSES)
    weights = []
    print("\nTraining class counts:")
    for cls in config.CLASS_LIST:
        print(f"  {str(cls):8s}: {int(counts.get(cls, 0))}")

    for cls in config.CLASS_LIST:
        count = int(counts.get(cls, 0))
        weights.append(total / (n_classes * count) if count else 0.0)

    result = torch.tensor(weights, dtype=torch.float32, device=device)
    print("Class weights:", [round(float(x), 4) for x in result.detach().cpu()])
    return result


def autocast_context(device, enabled):
    # AMP is enabled only on CUDA. bfloat16 is not assumed because the
    # intended GPU is an RTX 3050 laptop; float16 + GradScaler is safer here.
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(enabled and device.type == "cuda"),
    )


@torch.no_grad()
def evaluate(model, loader, device, criterion, use_amp):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    for images, features, labels in tqdm(loader, desc="  evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            logits = model(images, features)
            loss = criterion(logits.float(), labels)

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total += bs
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    if total == 0:
        raise ValueError("Evaluation DataLoader contains zero samples.")

    return (
        total_loss / total,
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        all_preds,
        all_labels,
    )


def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler,
    use_amp, accumulation_steps, grad_clip
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, correct, total = 0.0, 0, 0
    num_batches = len(loader)
    if num_batches == 0:
        raise ValueError("Training DataLoader contains zero samples.")

    progress = tqdm(loader, desc="  train", leave=False)
    for batch_idx, (images, features, labels) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Scale by the actual size of the current accumulation group so the
        # final, possibly shorter group is not underweighted.
        group_start = (batch_idx // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, num_batches - group_start)

        with autocast_context(device, use_amp):
            logits = model(images, features)
            loss = criterion(logits.float(), labels)
            scaled_loss = loss / group_size

        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        bs = labels.size(0)
        total_loss += float(loss.detach().item()) * bs
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += bs

        end_group = ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == num_batches)
        if end_group:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        progress.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / total, correct / total


def save_curves(history, path):
    if not history:
        return
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="Train")
    axes[0].plot(df["epoch"], df["val_loss"], marker="o", label="Validation")
    axes[0].set_title("Ensemble Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"], marker="o", label="Train accuracy")
    axes[1].plot(df["epoch"], df["val_acc"], marker="o", label="Validation accuracy")
    axes[1].plot(df["epoch"], df["val_macro_f1"], marker="s", label="Validation macro-F1")
    axes[1].set_title("Ensemble Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(df["epoch"], df["learning_rate"], marker="o")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, classes, path):
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("ConvNeXt + MaxViT Ensemble — Test Confusion Matrix")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_gpu_memory():
    if not torch.cuda.is_available():
        return
    mib = 1024 ** 2
    print("\nGPU memory:")
    print(f"  Allocated      : {torch.cuda.memory_allocated() / mib:.1f} MiB")
    print(f"  Reserved       : {torch.cuda.memory_reserved() / mib:.1f} MiB")
    print(f"  Peak allocated : {torch.cuda.max_memory_allocated() / mib:.1f} MiB")
    print(f"  Peak reserved  : {torch.cuda.max_memory_reserved() / mib:.1f} MiB")


def main():
    start_time = time.perf_counter()
    set_seeds()
    output_dir, log_dir, plots_dir = make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg("USE_AMP", True) and device.type == "cuda")

    # Both complete backbones are trained jointly. This is substantially more
    # memory intensive than either baseline alone. Start at batch=1 on a 4-GB
    # GPU; accumulation increases effective batch size, not activation memory.
    batch_size = int(cfg("ENSEMBLE_BATCH_SIZE", 4))
    accumulation_steps = int(cfg("ENSEMBLE_GRADIENT_ACCUMULATION_STEPS", 2))
    epochs = int(cfg("ENSEMBLE_EPOCHS", 50))
    patience = int(cfg("ENSEMBLE_EARLY_STOPPING_PATIENCE", 8))
    learning_rate = float(cfg("ENSEMBLE_LEARNING_RATE", 5e-5))
    weight_decay = float(cfg("ENSEMBLE_WEIGHT_DECAY", 1e-4))
    label_smoothing = float(cfg("ENSEMBLE_LABEL_SMOOTHING", 0.05))
    scheduler_factor = float(cfg("ENSEMBLE_LR_SCHEDULER_FACTOR", 0.5))
    scheduler_patience = int(cfg("ENSEMBLE_LR_SCHEDULER_PATIENCE", 3))
    min_lr = float(cfg("ENSEMBLE_LR_SCHEDULER_MIN_LR", 1e-6))
    grad_clip = float(cfg("ENSEMBLE_GRAD_CLIP", 1.0))

    if batch_size < 1 or accumulation_steps < 1:
        raise ValueError("Batch size and gradient accumulation steps must be >= 1.")

    print("=" * 72)
    print("CENTRALIZED CONVNEXT-TINY + MAXVIT-TINY ENSEMBLE TRAINING")
    print("=" * 72)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
        torch.cuda.reset_peak_memory_stats()
    print(f"Classes: {config.CLASS_LIST}")
    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Handcrafted feature dimension: {config.HANDCRAFTED_FEATURE_DIM}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {accumulation_steps}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")
    print(f"Epochs / early stopping patience: {epochs} / {patience}")
    print(f"Learning rate / weight decay: {learning_rate} / {weight_decay}")
    print(f"AMP: {use_amp}")

    print("\nBuilding data loaders...")
    train_loader = build_centralized_train_loader(train_features=None)
    if train_loader.batch_size != batch_size:
        train_loader = dataset_module.create_loader(
            dataframe=train_loader.dataset.df,
            feature_lookup=train_loader.dataset.feature_lookup,
            shuffle=True,
            batch_size=batch_size,
            training=True,
        )
    val_loader = build_eval_loader("val")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    print("\nBuilding joint ensemble...")
    model = EnsembleClassifier(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=config.HANDCRAFTED_FEATURE_DIM,
        pretrained=True,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print("Fusion: equal-weight mean of ConvNeXt and MaxViT logits; both models train jointly.")

    class_weights = compute_class_weights(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint_path = output_dir / "centralized_convnext_maxvit_ensemble_best.pt"
    log_path = log_dir / "centralized_convnext_maxvit_ensemble_training_log.csv"
    curve_path = plots_dir / "centralized_convnext_maxvit_ensemble_training_curves.png"
    report_path = log_dir / "centralized_convnext_maxvit_ensemble_classification_report.txt"
    per_class_path = log_dir / "centralized_convnext_maxvit_ensemble_per_class_f1.csv"
    cm_path = plots_dir / "centralized_convnext_maxvit_ensemble_confusion_matrix.png"

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            use_amp, accumulation_steps, grad_clip
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, device, criterion, use_amp
        )

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": current_lr,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(log_path, index=False)

        print(
            f"\nEpoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}, macro-F1 {val_f1:.4f} | "
            f"LR {current_lr:.2e} | {row['epoch_seconds']:.1f}s"
        )
        if current_lr < old_lr:
            print(f"  Learning rate reduced: {old_lr:.2e} -> {current_lr:.2e}")

        # Macro-F1 is the checkpoint-selection metric because HAM10000 is
        # class-imbalanced. Validation only is used for model selection.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_val_f1,
                    "best_val_accuracy": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "class_list": list(config.CLASS_LIST),
                    "num_classes": int(config.NUM_CLASSES),
                    "handcrafted_feature_dim": int(config.HANDCRAFTED_FEATURE_DIM),
                    "fusion": "equal_weight_logit_mean",
                },
                checkpoint_path,
            )
            print(f"  Saved new best checkpoint (validation macro-F1={best_val_f1:.4f}).")
        else:
            no_improve += 1
            print(f"  No validation macro-F1 improvement: {no_improve}/{patience}")

        if device.type == "cuda":
            print(
                f"  Peak GPU allocated: "
                f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
            )

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if not checkpoint_path.exists():
        raise RuntimeError("No best checkpoint was saved; training did not complete successfully.")

    # Restore the best validation checkpoint before the single held-out test evaluation.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loader = build_eval_loader("test")
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device, criterion, use_amp
    )

    report = classification_report(
        test_labels,
        test_preds,
        labels=list(range(config.NUM_CLASSES)),
        target_names=[str(x) for x in config.CLASS_LIST],
        digits=4,
        zero_division=0,
    )
    report_path.write_text(
        "ConvNeXt-Tiny + MaxViT-Tiny Hybrid Ensemble — Held-out Test Report\n"
        f"Best epoch: {best_epoch}\n"
        f"Best validation accuracy: {best_val_acc:.6f}\n"
        f"Best validation macro-F1: {best_val_f1:.6f}\n"
        f"Test loss: {test_loss:.6f}\n"
        f"Test accuracy: {test_acc:.6f}\n"
        f"Test macro-F1: {test_f1:.6f}\n\n"
        + report,
        encoding="utf-8",
    )

    per_class = f1_score(
        test_labels,
        test_preds,
        labels=list(range(config.NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    pd.DataFrame({"class": config.CLASS_LIST, "f1_score": per_class}).to_csv(
        per_class_path, index=False
    )
    save_curves(history, curve_path)
    save_confusion_matrix(test_labels, test_preds, config.CLASS_LIST, cm_path)

    print("\n" + "=" * 72)
    print("FINAL ENSEMBLE RESULTS")
    print("=" * 72)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro-F1: {test_f1:.4f}")
    print("\nClassification report:\n")
    print(report)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training log: {log_path}")
    print(f"Classification report: {report_path}")
    print(f"Per-class F1: {per_class_path}")
    print(f"Training curves: {curve_path}")
    print(f"Confusion matrix: {cm_path}")
    print_gpu_memory()
    print(f"\nTotal elapsed: {(time.perf_counter() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA ran out of memory while jointly training both full backbones. "
            "Try ENSEMBLE_BATCH_SIZE=1, keep AMP enabled, close other GPU processes, "
            "and ensure no other training process is using the GPU. Gradient "
            "accumulation does not reduce the memory needed for one forward/backward pass."
        ) from exc
