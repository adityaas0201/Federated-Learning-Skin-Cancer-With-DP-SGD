"""
09H_federated_client_fedprox_ensemble.py

FedProx local client for the ConvNeXt-Tiny + MaxViT-Tiny ensemble.
The server still performs sample-weighted aggregation; this module adds
FedProx's proximal penalty to local optimization when global_state is supplied.

Run from project root for a one-epoch smoke test:
    python src/09H_federated_client_fedprox_ensemble.py
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = bool(getattr(config, "USE_AMP", True) and DEVICE.type == "cuda")
LOCAL_EPOCHS = int(getattr(config, "LOCAL_EPOCHS", 3))
LOCAL_LR = float(getattr(config, "FEDERATED_LEARNING_RATE",
                         getattr(config, "LEARNING_RATE", 1e-4)))
WEIGHT_DECAY = float(getattr(config, "FEDERATED_WEIGHT_DECAY",
                             getattr(config, "WEIGHT_DECAY", 1e-4)))
LABEL_SMOOTHING = float(getattr(config, "LABEL_SMOOTHING", 0.05))
GRAD_CLIP = float(getattr(config, "FEDERATED_GRADIENT_CLIP", 1.0))
FEDPROX_MU = float(getattr(config, "FEDPROX_MU", 0.01))
if FEDPROX_MU < 0:
    raise ValueError("FEDPROX_MU must be non-negative.")


def _load_module(filename: str, module_name: str):
    path = SRC_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


dataset_module = _load_module("06_dataset.py", "dataset_module_09g")
ensemble_module = _load_module("07G_ensemble_model.py", "ensemble_module_09g")
EnsembleClassifier = ensemble_module.ConvNeXtMaxViTEnsemble


def _client_csv(client_id: int) -> Path:
    return Path(config.OUTPUT_DIR) / "client_partitions" / f"client_{client_id}.csv"


def load_client_dataframe(client_id: int) -> pd.DataFrame:
    path = _client_csv(client_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Client partition not found: {path}\n"
            "Run your Dirichlet partition script first."
        )
    df = pd.read_csv(path)
    required = {"image_id", "filepath", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"Client {client_id} has zero samples.")
    return df


def build_federated_client_loader(client_id: int, train_features=None,
                                  batch_size: int = 4):
    """Build the existing project's client dataset and force a small batch."""
    if train_features is None:
        scaler = dataset_module.load_scaler()
        feature_path = Path(config.HANDCRAFTED_FEATURE_DIR) / "train_features.npz"
        train_features = dataset_module.prepare_scaled_features(
            "train", str(feature_path), scaler
        )

    client_df = load_client_dataframe(client_id)
    original_loader = dataset_module.build_client_loader(
        client_id=client_id,
        train_features=train_features,
    )
    if len(original_loader.dataset) != len(client_df):
        raise ValueError(
            f"Client {client_id} mismatch: CSV={len(client_df)}, "
            f"dataset={len(original_loader.dataset)}"
        )

    # Keep the project's dataset/transforms, but use batch_size=4 for the
    # two-backbone model on a 4-GB GPU.
    return DataLoader(
        original_loader.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )


def get_model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_model_state(model, state_dict):
    model.load_state_dict(state_dict, strict=True)
    return model


def compute_proximal_term(model, global_params, device):
    """Compute sum ||w - w_global||^2 over trainable parameters in FP32."""
    proximal_term = torch.zeros((), device=device, dtype=torch.float32)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name in global_params:
            proximal_term = proximal_term + torch.sum(
                (parameter.float() - global_params[name].float()).square()
            )
    return proximal_term


def train_client(client_id: int, global_state=None, train_features=None,
                 local_epochs: int = 1, device=DEVICE, batch_size: int = 4,
                 mu: float | None = None):
    """Train one client with FedProx and return its state + sample count.

    FedProx is active when global_state is supplied and mu > 0. The global
    parameter reference remains fixed for the entire local training call.
    """
    if mu is None:
        mu = FEDPROX_MU
    mu = float(mu)
    if mu < 0:
        raise ValueError("FedProx mu must be non-negative.")
    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1.")

    df = load_client_dataframe(client_id)
    loader = build_federated_client_loader(
        client_id, train_features=train_features, batch_size=batch_size
    )
    model = EnsembleClassifier(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=config.HANDCRAFTED_FEATURE_DIM,
        pretrained=(global_state is None),
    ).to(device)

    if global_state is not None:
        load_model_state(model, global_state)

    # Snapshot the server's trainable parameters once. Do not update this
    # reference during local epochs: FedProx penalizes drift from round start.
    global_params = {}
    if global_state is not None and mu > 0.0:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if name not in global_state:
                    raise KeyError(
                        f"Global state is missing trainable parameter: {name}"
                    )
                global_params[name] = global_state[name].detach().to(
                    device=device, dtype=torch.float32
                ).clone()
        if not global_params:
            raise RuntimeError("FedProx could not find global trainable parameters.")
    elif global_state is None:
        print("WARNING: no global_state supplied; this is a standalone smoke test, "
              "so the FedProx proximal penalty is inactive.")

    # Use capped square-root inverse-frequency weights. The previous full
    # inverse-frequency formula could assign very large weights to rare local
    # classes, producing extreme losses on highly non-IID clients. Disable
    # label smoothing here because classes absent from a client have zero weight.
    counts = df["label"].value_counts()
    present_counts = [int(counts.get(cls, 0)) for cls in config.CLASS_LIST]
    if not any(present_counts):
        raise ValueError(f"Client {client_id} has no recognized class labels.")
    max_count = max(present_counts)
    weights = []
    for count in present_counts:
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(min(5.0, max(0.25, (max_count / count) ** 0.5)))
    # Normalize weights so the average weight over this client's examples is 1.
    weighted_mean = sum(
        present_counts[i] * weights[i] for i in range(len(present_counts))
    ) / max(len(df), 1)
    if weighted_mean > 0:
        weights = [w / weighted_mean for w in weights]
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.0,
    )
    label_to_idx = {name: idx for idx, name in enumerate(config.CLASS_LIST)}
    unknown_labels = sorted(set(df["label"].astype(str)) - set(label_to_idx))
    if unknown_labels:
        raise ValueError(
            f"Client {client_id} contains labels not in config.CLASS_LIST: {unknown_labels}"
        )
    print(f"Client {client_id} label distribution: {dict(counts)}")
    print(f"Client {client_id} class weights (order={config.CLASS_LIST}): "
          f"{[round(w, 3) for w in weights]}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LOCAL_LR, weight_decay=WEIGHT_DECAY)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    print("\n" + "=" * 68)
    print(f"ENSEMBLE CLIENT {client_id} CHECK / LOCAL TRAINING")
    print("=" * 68)
    print(f"Device: {device} | samples: {len(loader.dataset)} | batch: {batch_size}")
    print(f"Local epochs: {local_epochs} | AMP: {USE_AMP}")
    print(f"Algorithm: FedProx | mu: {mu:.6g} | proximal penalty active: {bool(global_params)}")
    print("Fusion: probability-level ConvNeXt + MaxViT ensemble")

    history = []
    start = time.perf_counter()
    for epoch in range(1, local_epochs + 1):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        convnext_correct, maxvit_correct = 0, 0
        all_preds, all_labels = [], []
        progress = tqdm(loader, desc=f"Client {client_id} epoch {epoch}/{local_epochs}")
        for images, features, labels in progress:
            images = images.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    logits, aux = model(images, features, return_aux=True)
                    conv_logits = aux["convnext_logits"]
                    maxvit_logits = aux["maxvit_logits"]
                    for branch_name, branch_logits in (
                        ("ConvNeXt", conv_logits), ("MaxViT", maxvit_logits),
                        ("ensemble", logits),
                    ):
                        if not torch.isfinite(branch_logits).all():
                            raise RuntimeError(
                                f"Client {client_id} produced NaN/Inf {branch_name} logits."
                            )
                    # Auxiliary branch supervision: each backbone must learn to
                    # classify correctly on its own; the fused logits are used
                    # for the reported ensemble metrics. This avoids one branch
                    # dominating the gradient simply because its logit scale differs.
                    loss_conv = criterion(conv_logits.float(), labels)
                    loss_max = criterion(maxvit_logits.float(), labels)
                    classification_loss = 0.5 * (loss_conv + loss_max)
                    proximal_term = compute_proximal_term(model, global_params, device)
                    loss = classification_loss + (mu / 2.0) * proximal_term
                if epoch == 1 and seen == 0:
                    print(
                        f"Client {client_id} first-batch diagnostic: "
                        f"labels={torch.bincount(labels, minlength=config.NUM_CLASSES).tolist()}, "
                        f"ConvNeXt loss={loss_conv.detach().float().item():.4f}, "
                        f"MaxViT loss={loss_max.detach().float().item():.4f}, "
                        f"ensemble loss={criterion(logits.float(), labels).detach().float().item():.4f}, "
                        f"proximal penalty={(mu / 2.0 * proximal_term.detach()).item():.6f}, "
                        f"ConvNeXt logits=[{conv_logits.detach().float().min().item():.3f},"
                        f"{conv_logits.detach().float().max().item():.3f}], "
                        f"MaxViT logits=[{maxvit_logits.detach().float().min().item():.3f},"
                        f"{maxvit_logits.detach().float().max().item():.3f}]"
                    )
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                logits, aux = model(images, features, return_aux=True)
                conv_logits = aux["convnext_logits"]
                maxvit_logits = aux["maxvit_logits"]
                for branch_name, branch_logits in (
                    ("ConvNeXt", conv_logits), ("MaxViT", maxvit_logits),
                    ("ensemble", logits),
                ):
                    if not torch.isfinite(branch_logits).all():
                        raise RuntimeError(
                            f"Client {client_id} produced NaN/Inf {branch_name} logits."
                        )
                loss_conv = criterion(conv_logits.float(), labels)
                loss_max = criterion(maxvit_logits.float(), labels)
                classification_loss = 0.5 * (loss_conv + loss_max)
                proximal_term = compute_proximal_term(model, global_params, device)
                loss = classification_loss + (mu / 2.0) * proximal_term
                if epoch == 1 and seen == 0:
                    print(
                        f"Client {client_id} first-batch diagnostic: "
                        f"labels={torch.bincount(labels, minlength=config.NUM_CLASSES).tolist()}, "
                        f"ConvNeXt loss={loss_conv.detach().item():.4f}, "
                        f"MaxViT loss={loss_max.detach().item():.4f}, "
                        f"ensemble loss={criterion(logits.float(), labels).detach().item():.4f}, "
                        f"proximal penalty={(mu / 2.0 * proximal_term.detach()).item():.6f}, "
                        f"ConvNeXt logits=[{conv_logits.detach().min().item():.3f},"
                        f"{conv_logits.detach().max().item():.3f}], "
                        f"MaxViT logits=[{maxvit_logits.detach().min().item():.3f},"
                        f"{maxvit_logits.detach().max().item():.3f}]"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            if logits.ndim != 2 or logits.shape != (labels.shape[0], config.NUM_CLASSES):
                raise RuntimeError(f"Unexpected ensemble output shape: {tuple(logits.shape)}")
            if not torch.isfinite(logits).all():
                raise RuntimeError("Ensemble produced NaN/Inf logits.")
            batch_n = labels.size(0)
            total_loss += float(loss.detach().item()) * batch_n
            preds = logits.detach().argmax(dim=1)
            conv_preds = conv_logits.detach().argmax(dim=1)
            maxvit_preds = maxvit_logits.detach().argmax(dim=1)
            correct += int((preds == labels).sum().item())
            convnext_correct += int((conv_preds == labels).sum().item())
            maxvit_correct += int((maxvit_preds == labels).sum().item())
            seen += batch_n
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(seen, 1):.3f}")

        if seen == 0:
            raise RuntimeError(f"Client {client_id} loader returned no batches.")
        row = {
            "epoch": epoch,
            "loss": total_loss / seen,
            "accuracy": accuracy_score(all_labels, all_preds),
            "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "convnext_accuracy": convnext_correct / max(seen, 1),
            "maxvit_accuracy": maxvit_correct / max(seen, 1),
        }
        history.append(row)
        print(f"Epoch {epoch}: total FedProx loss={row['loss']:.4f}, "
              f"ensemble accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}, "
              f"ConvNeXt accuracy={row['convnext_accuracy']:.4f}, "
              f"MaxViT accuracy={row['maxvit_accuracy']:.4f}")

    elapsed = time.perf_counter() - start
    state = get_model_state(model)
    result = {
        "client_id": client_id,
        "state_dict": state,
        "num_samples": len(loader.dataset),
        "train_loss": history[-1]["loss"],
        "train_accuracy": history[-1]["accuracy"],
        "train_macro_f1": history[-1]["macro_f1"],
        "local_epochs": local_epochs,
        "history": history,
    }
    print(f"\nCLIENT {client_id} FEDPROX CHECK PASSED")
    print(f"Returned state tensors: {len(state)} | elapsed: {elapsed / 60:.2f} min")
    if torch.cuda.is_available():
        print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
        print(f"GPU reserved : {torch.cuda.memory_reserved() / 1024**2:.0f} MB")
    del model, optimizer, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    # Standalone smoke test only: without global_state the proximal penalty
    # is inactive. The federated driver must pass the server's global_state.
    seed = int(getattr(config, "RANDOM_SEED", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    CLIENT_ID = 0
    train_client(client_id=CLIENT_ID, local_epochs=1, batch_size=4)
