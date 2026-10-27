"""
11H_train_federated_fedprox_ensemble.py
--------------------------------
Step 11H: end-to-end FedProx training with sample-weighted server aggregation for the
ConvNeXt-Tiny + MaxViT-Tiny hybrid ensemble.

Uses the existing project modules:
  06_dataset.py
  07G_ensemble_model.py
  09H_federated_client_fedprox_ensemble.py
  10G_fedavg_server_ensemble.py

The client batch size is 4. Evaluation uses batch size 1 to reduce VRAM
usage on a 4-GB GPU. This script runs clients sequentially, aggregates
their ensemble state dictionaries with sample-weighted averaging, selects the best round by
validation Macro-F1, and evaluates the held-out test set once at the end.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

import config


def load_module(filename: str, module_name: str):
    path = SRC_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Required project file not found:\n  {path}\n"
            "Keep the Step 11H script beside the other src files."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


client_module = load_module(
    "09H_federated_client_fedprox_ensemble.py", "client_module_11h"
)
server_module = load_module(
    "10G_fedavg_server_ensemble.py", "server_module_11h"
)
dataset_module = load_module("06_dataset.py", "dataset_module_11h")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 1
NUM_CLASSES = int(config.NUM_CLASSES)
CLASS_NAMES = list(config.CLASS_LIST)
FEATURE_DIM = int(config.HANDCRAFTED_FEATURE_DIM)
SEED = int(getattr(config, "RANDOM_SEED", 42))
# Requested experiment settings: 10 local epochs/client x up to 20 rounds.
# FedProx mu is adapted automatically from validation-loss trends after each round.
LOCAL_EPOCHS = 3
NUM_ROUNDS = 10
INITIAL_FEDPROX_MU = float(getattr(config, "FEDPROX_MU", 0.01))
FEDPROX_MU_MIN = float(getattr(config, "FEDPROX_MU_MIN", 0.001))
FEDPROX_MU_MAX = float(getattr(config, "FEDPROX_MU_MAX", 1.0))
FEDPROX_MU_INCREASE = float(getattr(config, "FEDPROX_MU_INCREASE_FACTOR", 1.25))
FEDPROX_MU_DECREASE = float(getattr(config, "FEDPROX_MU_DECREASE_FACTOR", 1.10))
FEDPROX_LOSS_TOLERANCE = float(getattr(config, "FEDPROX_LOSS_TOLERANCE", 0.02))
PATIENCE = int(getattr(config, "FED_EARLY_STOPPING_PATIENCE", 6))
MIN_DELTA = float(getattr(config, "FED_MIN_DELTA", 1e-4))
CLIENT_LR = float(
    getattr(config, "FEDERATED_LEARNING_RATE",
            getattr(config, "LEARNING_RATE", 1e-4))
)
CLIENTS_PER_ROUND = getattr(
    config, "FED_CLIENTS_PER_ROUND",
    getattr(config, "CLIENTS_PER_ROUND", None)
)
NUM_CLIENTS = int(
    getattr(config, "NUM_CLIENTS",
            getattr(config, "NUM_FEDERATED_CLIENTS", 10))
)
if CLIENTS_PER_ROUND is None:
    CLIENTS_PER_ROUND = NUM_CLIENTS
CLIENTS_PER_ROUND = int(CLIENTS_PER_ROUND)
USE_AMP = bool(getattr(config, "USE_AMP", True) and DEVICE.type == "cuda")


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed: {SEED}")


def get_client_ids():
    ids = list(range(NUM_CLIENTS))
    missing = [
        i for i in ids
        if not (Path(config.OUTPUT_DIR) / "client_partitions" / f"client_{i}.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing client partition CSV(s): "
            + ", ".join(str(i) for i in missing)
            + f"\nExpected under: {Path(config.OUTPUT_DIR) / 'client_partitions'}"
            + "\nCheck NUM_CLIENTS / NUM_FEDERATED_CLIENTS in config.py."
        )
    return ids


def build_eval_loader(split: str):
    base = dataset_module.build_eval_loader(split)
    return DataLoader(
        base.dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )


@torch.no_grad()
def evaluate(model, loader, split_name: str):
    model.eval()
    total_loss = 0.0
    all_predictions, all_labels = [], []
    # Unweighted CE is used for evaluation so the metric is comparable
    # across rounds and independent of any one client's class distribution.
    criterion = torch.nn.CrossEntropyLoss()
    for images, features, labels in tqdm(loader, desc=f"{split_name} evaluation", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        features = features.to(DEVICE, non_blocking=True).float()
        labels = labels.to(DEVICE, non_blocking=True).long()
        logits = model(images, features)
        if logits.shape != (labels.shape[0], NUM_CLASSES):
            raise RuntimeError(f"Expected logits {(labels.shape[0], NUM_CLASSES)}, got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"NaN/Inf logits encountered during {split_name} evaluation.")
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    if not all_labels:
        raise RuntimeError(f"{split_name} evaluation loader is empty.")
    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
        "predictions": all_predictions,
        "labels": all_labels,
    }


def save_training_plot(df: pd.DataFrame, path: Path):
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df["round"], df["val_loss"], marker="o", label="Validation loss")
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Federated round")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(df["round"], df["val_accuracy"], marker="o", label="Validation accuracy")
    axes[1].plot(df["round"], df["val_macro_f1"], marker="s", label="Validation Macro-F1")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Federated round")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(labels, predictions, path: Path):
    ids = list(range(NUM_CLASSES))
    cm = confusion_matrix(labels, predictions, labels=ids)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)
    ax.set_xticks(ids, labels=CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(ids, labels=CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("FedProx Ensemble — Held-out Test Confusion Matrix")
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    color="white" if cm[r, c] > threshold else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    start = time.perf_counter()
    set_seeds()

    output_dir = Path(config.OUTPUT_DIR)
    model_dir = output_dir / "models"
    log_dir = output_dir / "training_logs"
    plots_dir = output_dir / "plots"
    for folder in (model_dir, log_dir, plots_dir):
        folder.mkdir(parents=True, exist_ok=True)

    tag = str(getattr(config, "FEDPROX_ENSEMBLE_EXPERIMENT_TAG", "11H_fedprox_adaptive_ensemble_20rounds"))
    best_path = model_dir / f"{tag}_global_best.pt"
    final_path = model_dir / f"{tag}_global_final.pt"
    log_path = log_dir / f"{tag}_training_log.csv"
    report_path = log_dir / f"{tag}_test_classification_report.txt"
    per_class_path = log_dir / f"{tag}_test_per_class_f1.csv"
    curve_path = plots_dir / f"{tag}_training_curve.png"
    cm_path = plots_dir / f"{tag}_test_confusion_matrix.png"

    print("\n" + "=" * 72)
    print("STEP 11H — CONVNEXT + MAXVIT ENSEMBLE FEDERATED TRAINING (FEDPROX)")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    print(f"Client batch size: {BATCH_SIZE}")
    print(f"Evaluation batch size: {EVAL_BATCH_SIZE}")
    print(f"Clients: {NUM_CLIENTS} | clients per round: {CLIENTS_PER_ROUND}")
    print(f"Rounds: {NUM_ROUNDS} | local epochs/client/round: {LOCAL_EPOCHS}")
    print(
        f"Configured client LR: {CLIENT_LR:.2e} | initial FedProx mu: {INITIAL_FEDPROX_MU:.4f} "
        f"(auto range {FEDPROX_MU_MIN:.4f}–{FEDPROX_MU_MAX:.4f}) | AMP: {USE_AMP}"
    )
    print(f"Classes: {CLASS_NAMES} | handcrafted feature dimension: {FEATURE_DIM}")

    if NUM_ROUNDS < 1 or LOCAL_EPOCHS < 1:
        raise ValueError("NUM_ROUNDS and LOCAL_EPOCHS must be >= 1.")
    if INITIAL_FEDPROX_MU <= 0:
        raise ValueError("INITIAL_FEDPROX_MU must be > 0 for active FedProx.")
    if not (0 < FEDPROX_MU_MIN <= INITIAL_FEDPROX_MU <= FEDPROX_MU_MAX):
        raise ValueError("Require 0 < FEDPROX_MU_MIN <= initial mu <= FEDPROX_MU_MAX.")
    if FEDPROX_MU_INCREASE <= 1.0 or FEDPROX_MU_DECREASE <= 1.0:
        raise ValueError("FedProx mu increase/decrease factors must be > 1.")
    if FEDPROX_LOSS_TOLERANCE < 0:
        raise ValueError("FEDPROX_LOSS_TOLERANCE must be >= 0.")
    if not 1 <= CLIENTS_PER_ROUND <= NUM_CLIENTS:
        raise ValueError("Clients per round must be between 1 and NUM_CLIENTS.")

    client_ids = get_client_ids()
    val_loader = build_eval_loader("val")
    test_loader = build_eval_loader("test")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Fresh ImageNet-pretrained initialization, matching the centralized
    # experiment's initialization policy; do not initialize from its checkpoint.
    global_model = server_module.build_global_model(DEVICE)
    initial_path = Path(server_module.INITIAL_MODEL_PATH)
    server_module.save_initial_global_model(global_model)

    best_f1 = -1.0
    best_round = 0
    rounds_without_improvement = 0
    rows = []
    total_samples_per_round = []
    current_mu = INITIAL_FEDPROX_MU
    previous_val_loss = None

    for round_number in range(1, NUM_ROUNDS + 1):
        round_start = time.perf_counter()
        if CLIENTS_PER_ROUND >= len(client_ids):
            selected = list(client_ids)
        else:
            rng = random.Random(SEED + round_number)
            selected = sorted(rng.sample(client_ids, CLIENTS_PER_ROUND))

        print("\n" + "-" * 72)
        print(f"FEDERATED ROUND {round_number}/{NUM_ROUNDS}")
        print(f"Selected clients: {selected}")
        print(f"FedProx mu for this round: {current_mu:.6f}")
        print("-" * 72)

        global_state = server_module.state_dict_to_cpu(global_model.state_dict())
        client_states, sample_counts = [], []
        client_losses, client_accuracies, client_macro_f1s = [], [], []

        for client_id in selected:
            print(f"\n>>> Training client {client_id}")
            try:
                result = client_module.train_client(
                    client_id=client_id,
                    global_state=global_state,
                    local_epochs=LOCAL_EPOCHS,
                    device=DEVICE,
                    batch_size=BATCH_SIZE,
                    mu=current_mu,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"CUDA out-of-memory on client {client_id} in round {round_number}. "
                        "Try batch size 2 or 1 in 11H_train_federated_fedprox_ensemble.py."
                    ) from exc
                raise

            state = result["state_dict"]
            server_module.validate_state_dict(
                state,
                server_module.state_dict_to_cpu(global_model.state_dict()),
                client_id=client_id,
            )
            client_states.append(state)
            sample_counts.append(int(result["num_samples"]))
            client_losses.append(float(result["train_loss"]))
            client_accuracies.append(float(result["train_accuracy"]))
            client_macro_f1s.append(float(result["train_macro_f1"]))

            del result, state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # FedProx changes the local objective; server aggregation remains sample-weighted averaging.
        server_module.aggregate_client_updates(global_model, client_states, sample_counts)
        total_samples = sum(sample_counts)
        total_samples_per_round.append(total_samples)

        val = evaluate(global_model, val_loader, f"Round {round_number} validation")
        elapsed_round = time.perf_counter() - round_start
        row = {
            "round": round_number,
            "selected_clients": ",".join(map(str, selected)),
            "participating_clients": len(selected),
            "aggregated_samples": total_samples,
            "fedprox_mu": current_mu,
            "mean_client_loss": float(np.mean(client_losses)),
            "mean_client_accuracy": float(np.mean(client_accuracies)),
            "mean_client_macro_f1": float(np.mean(client_macro_f1s)),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "round_seconds": elapsed_round,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_path, index=False)

        print(f"\nRound {round_number} summary:")
        print(f"  Aggregated samples : {total_samples}")
        print(f"  FedProx mu used    : {current_mu:.6f}")
        print(f"  Validation loss    : {val['loss']:.4f}")
        print(f"  Validation accuracy: {val['accuracy']:.4f}")
        print(f"  Validation Macro-F1: {val['macro_f1']:.4f}")
        print(f"  Round time         : {elapsed_round / 60:.2f} min")

        # Adaptive FedProx strength for the NEXT round:
        # if global validation loss materially worsens, strengthen the proximal
        # constraint; if it materially improves, relax it slightly. This is a
        # bounded heuristic, not a canonical requirement of the FedProx paper.
        if previous_val_loss is not None:
            if val["loss"] > previous_val_loss * (1.0 + FEDPROX_LOSS_TOLERANCE):
                current_mu = min(FEDPROX_MU_MAX, current_mu * FEDPROX_MU_INCREASE)
                print(f"  Adaptive mu update : validation loss worsened; next mu={current_mu:.6f}")
            elif val["loss"] < previous_val_loss * (1.0 - FEDPROX_LOSS_TOLERANCE):
                current_mu = max(FEDPROX_MU_MIN, current_mu / FEDPROX_MU_DECREASE)
                print(f"  Adaptive mu update : validation loss improved; next mu={current_mu:.6f}")
            else:
                print(f"  Adaptive mu update : validation loss stable; next mu={current_mu:.6f}")
        previous_val_loss = float(val["loss"])

        if val["macro_f1"] > best_f1 + MIN_DELTA:
            best_f1 = val["macro_f1"]
            best_round = round_number
            rounds_without_improvement = 0
            server_module.save_global_checkpoint(
                global_model,
                epoch=round_number,
                round_number=round_number,
                client_num_samples=sample_counts,
                val_macro_f1=val["macro_f1"],
                val_accuracy=val["accuracy"],
                path=str(best_path),
            )
            print(f"  New best checkpoint saved (Macro-F1={best_f1:.4f}).")
        else:
            rounds_without_improvement += 1
            print(f"  No improvement: {rounds_without_improvement}/{PATIENCE}")

        del client_states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if rounds_without_improvement >= PATIENCE:
            print(f"\nEarly stopping: validation Macro-F1 did not improve for {PATIENCE} rounds.")
            break

    if not best_path.is_file():
        raise FileNotFoundError(f"Best global checkpoint was not created: {best_path}")

    # Save last-round model separately, then restore the best validation checkpoint.
    server_module.save_global_checkpoint(
        global_model,
        epoch=rows[-1]["round"],
        round_number=rows[-1]["round"],
        client_num_samples=sample_counts,
        val_macro_f1=rows[-1]["val_macro_f1"],
        val_accuracy=rows[-1]["val_accuracy"],
        path=str(final_path),
    )
    log_df = pd.DataFrame(rows)
    log_df.to_csv(log_path, index=False)
    save_training_plot(log_df, curve_path)

    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    global_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"\nRestored best checkpoint from round {checkpoint['round']} "
          f"(validation Macro-F1={checkpoint['val_macro_f1']:.4f}).")

    # The held-out test set is evaluated only after model selection is complete.
    test = evaluate(global_model, test_loader, "Final test")
    report = classification_report(
        test["labels"],
        test["predictions"],
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    report_text = (
        "FEDERATED FEDPROX CONVNEXT + MAXVIT ENSEMBLE — TEST CLASSIFICATION REPORT\n"
        f"Local epochs per client per round: {LOCAL_EPOCHS}\n"
        f"Maximum federated rounds: {NUM_ROUNDS}\n"
        f"Initial FedProx mu: {INITIAL_FEDPROX_MU:.6f}\n"
        f"Final adaptive FedProx mu: {current_mu:.6f}\n\n"
        + report
        + f"\nTest loss: {test['loss']:.6f}\n"
        + f"Test accuracy: {test['accuracy']:.6f}\n"
        + f"Test Macro-F1: {test['macro_f1']:.6f}\n"
        + f"Best federated round: {best_round}\n"
        + f"Best validation Macro-F1: {best_f1:.6f}\n"
    )
    report_path.write_text(report_text, encoding="utf-8")
    per_class_scores = f1_score(
        test["labels"],
        test["predictions"],
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    pd.DataFrame({"class": CLASS_NAMES, "f1_score": per_class_scores}).to_csv(
        per_class_path, index=False
    )
    save_confusion_matrix(test["labels"], test["predictions"], cm_path)

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 72)
    print("STEP 11H COMPLETE")
    print("=" * 72)
    print(f"Best round          : {best_round}")
    print(f"Best validation F1  : {best_f1:.4f}")
    print(f"Final adaptive mu   : {current_mu:.6f}")
    print(f"Test loss           : {test['loss']:.4f}")
    print(f"Test accuracy       : {test['accuracy']:.4f}")
    print(f"Test Macro-F1       : {test['macro_f1']:.4f}")
    print(f"Total elapsed       : {elapsed / 60:.2f} minutes")
    print("\nSaved outputs:")
    for path in (best_path, final_path, log_path, report_path, per_class_path, curve_path, cm_path):
        print(f"  {path}")
    if torch.cuda.is_available():
        print(f"\nGPU peak allocated: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
        print(f"GPU peak reserved : {torch.cuda.max_memory_reserved() / 1024**2:.0f} MB")


if __name__ == "__main__":
    main()
