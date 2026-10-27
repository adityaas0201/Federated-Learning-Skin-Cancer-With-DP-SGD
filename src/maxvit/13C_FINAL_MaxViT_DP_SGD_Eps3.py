"""
13C - FINAL MAXVIT DP-SGD FEDERATED EXPERIMENT - EPSILON 3 - VERIFIED ALPHA-100.0

Design:
    MaxViT-Tiny V2 + GroupNorm task heads
    Opacus DP-SGD + Poisson sampling
    client-specific noise multipliers calibrated with Opacus RDP
    fixed global class-balanced CrossEntropyLoss
    3 local epochs x 10 federated rounds
    fixed FedProx mu=0.005
    smart aggregation + automatic FedYogi
    no WeightedRandomSampler / adaptive client LR / adaptive FedProx

IMPORTANT:
    Privacy accounting is maintained per client with a persistent
    Opacus PrivacyEngine across federated rounds. Empty Poisson batches
    are skipped and do not consume an optimizer/accountant step.
"""

import os

# Reduce CUDA allocator fragmentation on small-VRAM GPUs.
# Must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
import copy
import random
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)

from tqdm import tqdm
from torch.utils.data import DataLoader

from torchvision.ops import StochasticDepth

import opacus
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier


# =====================================================================
# PATH / PROJECT MODULES
# =====================================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")

if CONFIG_DIR not in sys.path:
    sys.path.append(CONFIG_DIR)

import config

# ---------------------------------------------------------------------
# EXPERIMENT PARTITION OVERRIDE
# ---------------------------------------------------------------------
# 06_dataset.py reads the partition configuration when it is imported.
# Set the intended alpha before importing the dataset module.
# IMPORTANT: if your project uses a pre-generated partition file, that
# partition must be regenerated with alpha=100.0; this assignment alone
# cannot rewrite an existing partition artifact.
TARGET_DIRICHLET_ALPHA = 100.0

# ---------------------------------------------------------------------
# VERIFIED 13C PARTITION LOCK
# ---------------------------------------------------------------------
# This experiment MUST read the pre-generated alpha_100.0 partition.
# Never fall back to outputs/client_partitions or regenerate a partition.
ALPHA100_DIR = os.path.abspath(
    os.path.join(config.OUTPUT_DIR, "alpha_sweep", "alpha_100.0")
)
ALPHA100_PARTITION_DIR = os.path.join(
    ALPHA100_DIR, "client_partitions"
)
EXPECTED_ALPHA100_CLIENT_SIZES = (1338, 1375, 1521, 1274, 1324, 1280)
EXPECTED_ALPHA100_TOTAL = 8112
for _alpha_name in ("DIRICHLET_ALPHA", "FED_DIRICHLET_ALPHA", "DIRICHLET_PARTITION_ALPHA"):
    if hasattr(config, _alpha_name):
        setattr(config, _alpha_name, TARGET_DIRICHLET_ALPHA)


def _load_module(filename, module_name):
    path = os.path.join(SRC_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required module not found:\n{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module:\n{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_module = _load_module("06_dataset.py", "dp13c_dataset")
model_module = _load_module("07F_model_maxvit_tiny.py", "dp13c_model")


def get_model_class(module):
    for name in [
        "MaxViTHybridClassifier",
        "MaxViTHybrid",
        "MaxViTClassifier",
        "HybridClassifier",
    ]:
        candidate = getattr(module, name, None)
        if isinstance(candidate, type):
            return candidate

    candidates = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        candidate = getattr(module, name)
        if isinstance(candidate, type):
            low = name.lower()
            if "maxvit" in low and "classifier" in low:
                candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]

    raise AttributeError("Could not locate the MaxViT hybrid classifier.")


MaxViTModel = get_model_class(model_module)


def get_client_loader_function(module):
    for name in [
        "build_client_train_loader",
        "build_federated_client_loader",
        "build_client_loader",
    ]:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    raise AttributeError("Could not locate the client train-loader function.")


build_client_train_loader = get_client_loader_function(dataset_module)
build_eval_loader = getattr(dataset_module, "build_eval_loader", None)
if build_eval_loader is None:
    raise AttributeError("06_dataset.py does not contain build_eval_loader().")


# =====================================================================
# REPRODUCIBILITY / DEVICE
# =====================================================================

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed: {seed}")


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU VRAM: {mem:.2f} GB")
    return device


# =====================================================================
# MODEL DP-COMPATIBILITY CONVERSION
# =====================================================================

def _group_count(num_channels, preferred=8):
    for groups in range(min(preferred, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def convert_bn_to_gn(module):
    converted = 0
    for name, child in list(module.named_children()):
        if isinstance(
            child,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm),
        ):
            groups = _group_count(child.num_features, 8)
            gn = nn.GroupNorm(
                num_groups=groups,
                num_channels=child.num_features,
                eps=child.eps,
                affine=child.affine,
            ).to(
                device=child.weight.device if child.affine else next(
                    child.parameters(), torch.empty(0)
                ).device,
                dtype=child.weight.dtype if child.affine else next(
                    child.parameters(), torch.empty(0)
                ).dtype,
            )
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
            converted += 1
        else:
            converted += convert_bn_to_gn(child)
    return converted


def disable_dp_incompatible_modules(module):
    relu_count = 0
    sd_count = 0
    dropout_count = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
            relu_count += 1
        elif isinstance(child, StochasticDepth):
            # MaxViT stochastic depth contains a random Bernoulli operation.
            # Disable it so Opacus per-sample gradient computation is deterministic.
            child.p = 0.0
            sd_count += 1
        elif isinstance(child, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            setattr(module, name, nn.Identity())
            dropout_count += 1
        else:
            r, s, d = disable_dp_incompatible_modules(child)
            relu_count += r
            sd_count += s
            dropout_count += d

    return relu_count, sd_count, dropout_count


def apply_partial_transfer_learning(model):
    """
    MaxViT partial transfer learning with Opacus compatibility.

    Strategy:
      1. Freeze the complete MaxViT image backbone.
      2. Unfreeze the final MaxViT block.
      3. Freeze RelativePositionalMultiHeadAttention modules inside
         that final block because Opacus 1.6 fast/gradient
         sampling does not support their trainable buffers.
      4. Keep the remaining final-block parameters trainable.
      5. Keep the handcrafted-feature branch and classifier head trainable.
    """
    if not hasattr(model, "backbone"):
        raise AttributeError("Model does not expose a 'backbone' attribute.")

    backbone = model.backbone

    if not hasattr(backbone, "features"):
        raise RuntimeError(
            "Validated MaxViT backbone.features was not found."
        )

    features = backbone.features

    if not isinstance(features, nn.Sequential):
        raise RuntimeError(
            "Validated MaxViT backbone.features is not nn.Sequential."
        )

    if len(features) < 2:
        raise RuntimeError(
            "MaxViT backbone.features must contain the stem and "
            "at least one MaxViT block."
        )

    # Freeze the complete image backbone first.
    for p in backbone.parameters():
        p.requires_grad = False

    # features[0] = stem; features[1:] = MaxViT blocks.
    final_block_index = len(features) - 1
    final_block = features[final_block_index]

    # Unfreeze the final MaxViT block.
    for p in final_block.parameters():
        p.requires_grad = True

    # Opacus 1.6 does not support these MaxViT attention modules as
    # trainable layers because they contain unsupported buffers.
    frozen_attention_modules = []
    frozen_attention_parameters = 0

    for module_name, module in final_block.named_modules():
        if module.__class__.__name__ == "RelativePositionalMultiHeadAttention":
            for p in module.parameters():
                if p.requires_grad:
                    frozen_attention_parameters += p.numel()
                p.requires_grad = False

            frozen_attention_modules.append(
                module_name if module_name else "<root>"
            )

    # Make sure the final block still has trainable parameters.
    final_block_trainable = sum(
        p.numel()
        for p in final_block.parameters()
        if p.requires_grad
    )

    if final_block_trainable <= 0:
        raise RuntimeError(
            "After freezing Opacus-incompatible attention modules, "
            "the final MaxViT block has no trainable parameters."
        )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    backbone_trainable = sum(
        p.numel()
        for p in backbone.parameters()
        if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())

    print("\nPartial transfer learning:")
    print(f"  Frozen parameters                    : {total - trainable:,}")
    print(f"  Trainable backbone parameters        : {backbone_trainable:,}")
    print(f"  Trainable final MaxViT block         : {final_block_trainable:,}")
    print(f"  Trainable total parameters           : {trainable:,}")
    print(f"  Final MaxViT block index             : {final_block_index}")
    print(f"  Frozen attention modules             : {len(frozen_attention_modules)}")
    print(f"  Frozen attention parameters          : {frozen_attention_parameters:,}")

    if frozen_attention_modules:
        print("\n  Opacus-incompatible attention modules:")
        for name in frozen_attention_modules:
            print(f"    - {name}")

    if backbone_trainable <= 0:
        raise RuntimeError("MaxViT backbone has no trainable parameters.")

    if trainable <= 0:
        raise RuntimeError("Model has no trainable parameters.")

    return (
        f"backbone.features.{final_block_index}",
        trainable,
        backbone_trainable,
    )

def build_dp_model(device, pretrained):
    model = MaxViTModel(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=config.HANDCRAFTED_FEATURE_DIM,
        pretrained=pretrained,
    ).to(device)

    bn_count = convert_bn_to_gn(model)
    relu_count, sd_count, dropout_count = disable_dp_incompatible_modules(model)

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        formatted = "\n".join(str(e) for e in errors[:20])
        raise RuntimeError(
            "Opacus ModuleValidator found compatibility errors after "
            "explicit DP conversion:\n" + formatted
        )

    # Partial transfer learning is applied after compatibility conversion.
    # Frozen parameters remain part of the model/state_dict but are excluded
    # from DP optimization.
    apply_partial_transfer_learning(model)

    return model, bn_count, relu_count, sd_count, dropout_count


# =====================================================================
# AGGRESSIVE CUDA / MEMORY CLEANUP
# =====================================================================

def aggressive_cuda_cleanup(tag=""):
    """
    Release Python references and cached CUDA blocks between clients/rounds.
    This is intentionally conservative for long Opacus hooks runs on GPUs
    with limited VRAM (e.g. RTX 3050 4 GB); hooks mode may use more memory than Ghost mode.
    """
    import gc

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    if tag:
        print(f"  CUDA cleanup completed: {tag}")


# =====================================================================
# CLIENT / DATASET HELPERS
# =====================================================================

def get_client_ids():
    if hasattr(config, "NUM_CLIENTS"):
        n = int(config.NUM_CLIENTS)
    elif hasattr(config, "NUM_FEDERATED_CLIENTS"):
        n = int(config.NUM_FEDERATED_CLIENTS)
    else:
        n = 10
    if n <= 0:
        raise ValueError("Number of clients must be > 0.")
    return list(range(n))


def validate_alpha100_partition(client_ids):
    """Hard-lock 13C to the verified alpha_100.0 client CSVs."""
    if len(client_ids) != 6 or list(client_ids) != [0, 1, 2, 3, 4, 5]:
        raise RuntimeError(
            "13C requires exactly six clients with IDs [0, 1, 2, 3, 4, 5]."
        )

    if not os.path.isdir(ALPHA100_PARTITION_DIR):
        raise FileNotFoundError(
            "13C REFUSED TO TRAIN: verified alpha_100.0 partition directory "
            f"does not exist:\n{ALPHA100_PARTITION_DIR}"
        )

    actual = []
    for cid in client_ids:
        path = os.path.join(ALPHA100_PARTITION_DIR, f"client_{cid}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"13C REFUSED TO TRAIN: missing verified partition file:\n{path}"
            )
        df = pd.read_csv(path)
        actual.append(len(df))

    actual_tuple = tuple(actual)
    if actual_tuple != EXPECTED_ALPHA100_CLIENT_SIZES:
        raise RuntimeError(
            "13C REFUSED TO TRAIN: alpha_100.0 client sizes do not match "
            "the verified partition.\n"
            f"Expected: {EXPECTED_ALPHA100_CLIENT_SIZES}\n"
            f"Found   : {actual_tuple}\n"
            f"Directory: {ALPHA100_PARTITION_DIR}"
        )

    if sum(actual_tuple) != EXPECTED_ALPHA100_TOTAL:
        raise RuntimeError(
            f"13C REFUSED TO TRAIN: expected total {EXPECTED_ALPHA100_TOTAL}, "
            f"found {sum(actual_tuple)}."
        )

    print("\nVERIFIED 13C PARTITION LOCK: PASSED")
    print(f"  Partition directory : {ALPHA100_PARTITION_DIR}")
    for cid, n in zip(client_ids, actual_tuple):
        print(f"  Client {cid}: {n} samples")
    print(f"  Total               : {sum(actual_tuple)}")


def call_base_client_loader(client_id):
    """Load a client ONLY from the locked alpha_100.0 directory."""
    original_output_dir = config.OUTPUT_DIR
    try:
        # 06_dataset.py resolves client CSVs as
        # config.OUTPUT_DIR/client_partitions/client_<id>.csv.
        # Temporarily point that lookup at alpha_100.0, then restore the
        # normal output root before any evaluation/output paths are used.
        config.OUTPUT_DIR = ALPHA100_DIR
        try:
            return build_client_train_loader(client_id)
        except TypeError:
            return build_client_train_loader(client_id=client_id)
    finally:
        config.OUTPUT_DIR = original_output_dir


def build_standard_client_loader(client_id, batch_size, dataset=None):
    # Reuse the same Dataset object across rounds when supplied. This avoids
    # Opacus' "new dataset object" warning and keeps the per-client logical
    # dataset identity stable for the persistent PrivacyEngine accountant.
    if dataset is None:
        base_loader = call_base_client_loader(client_id)
        dataset = base_loader.dataset

    if len(dataset) <= 0:
        raise ValueError(f"Client {client_id} contains zero samples.")

    # Preserve the Dataset object from 06_dataset.py, including its
    # existing on-the-fly augmentation. Do NOT use WeightedRandomSampler.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def collect_client_datasets(client_ids, batch_size):
    datasets = {}
    loaders = {}
    for cid in client_ids:
        loader = build_standard_client_loader(cid, batch_size)
        loaders[cid] = loader
        datasets[cid] = loader.dataset
    return datasets, loaders


# =====================================================================
# FIXED GLOBAL CLASS-BALANCED LOSS
# =====================================================================

def compute_global_class_weights(datasets, device):
    counts = {cls: 0 for cls in config.CLASS_LIST}

    for cid, dataset in datasets.items():
        if not hasattr(dataset, "df"):
            raise AttributeError(
                f"Client {cid} dataset must expose dataset.df."
            )
        if "label" not in dataset.df.columns:
            raise AttributeError(
                f"Client {cid} dataset dataframe must contain 'label'."
            )

        for cls in config.CLASS_LIST:
            counts[cls] += int((dataset.df["label"] == cls).sum())

    total = sum(counts.values())
    present = [cls for cls in config.CLASS_LIST if counts[cls] > 0]
    if total <= 0 or not present:
        raise ValueError("Could not compute global class counts.")

    power = float(getattr(config, "FED_CLASS_WEIGHT_POWER", 0.5))
    min_w = float(getattr(config, "FED_MIN_CLASS_WEIGHT", 0.25))
    max_w = float(getattr(config, "FED_MAX_CLASS_WEIGHT", 5.0))
    label_smoothing = float(
        getattr(config, "FED_LABEL_SMOOTHING",
                getattr(config, "LABEL_SMOOTHING", 0.02))
    )

    weights = []
    print("\nGlobal fixed class counts:")
    for cls in config.CLASS_LIST:
        print(f"  {cls:6s}: {counts[cls]}")

    for cls in config.CLASS_LIST:
        count = counts[cls]
        if count == 0:
            w = 1.0
        else:
            raw = total / (len(present) * count)
            w = raw ** power
            w = float(np.clip(w, min_w, max_w))
        weights.append(w)

    tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    print(
        f"\nGlobal fixed class-balanced weights "
        f"(power={power:.2f}, min={min_w:.2f}, max={max_w:.2f}):"
    )
    for cls, w in zip(config.CLASS_LIST, weights):
        print(f"  {cls:6s}: {w:.6f}")

    return tensor, counts, label_smoothing


def build_train_criterion(class_weights, label_smoothing):
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


def build_eval_criterion(label_smoothing):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


# =====================================================================
# PRIVACY CALIBRATION
# =====================================================================

def calibrate_client_sigmas(
    client_loaders,
    num_rounds,
    local_epochs,
    batch_size,
    target_epsilon,
    target_delta,
):
    """
    Calibrate each client's noise multiplier with Opacus's RDP
    noise-multiplier utility for the exact planned number of DP
    optimizer steps.

    The planned step count is:
        rounds * local_epochs * len(client_loader)

    Poisson sampling uses q = batch_size / N.
    Empty Poisson batches are skipped by the training loop, so the actual
    number of accountant-consuming optimizer steps cannot exceed this planned
    count. Therefore the calibrated sigma is conservative with respect to
    the planned horizon.

    The resulting sigma is intentionally client-specific because N
    differs strongly between clients.
    """
    sigmas = {}
    metadata = {}

    print("\n" + "=" * 72)
    print("DP-SGD CLIENT-SPECIFIC NOISE CALIBRATION")
    print("=" * 72)
    print(f"Target epsilon       : {target_epsilon:.6f}")
    print(f"Target delta         : {target_delta:.2e}")
    print(f"Physical batch size  : {batch_size}")
    print(f"Local epochs         : {local_epochs}")
    print(f"Federated rounds     : {num_rounds}")
    print("Accountant           : RDP")
    print("Sampling             : Poisson")

    for cid, loader in client_loaders.items():
        n = len(loader.dataset)
        # Opacus Poisson/UniformWithReplacement sampling uses this
        # expected sampling rate. The number of attempted batches is
        # ceil(N / physical_batch_size) for this loader configuration.
        q = min(1.0, batch_size / float(n))
        steps_per_epoch = max(1, int(np.ceil(n / float(batch_size))))
        planned_steps = num_rounds * local_epochs * steps_per_epoch

        sigma = get_noise_multiplier(
            target_epsilon=float(target_epsilon),
            target_delta=float(target_delta),
            sample_rate=float(q),
            steps=int(planned_steps),
            accountant="rdp",
            epsilon_tolerance=0.001,
        )

        sigmas[cid] = float(sigma)
        metadata[cid] = {
            "n": n,
            "sample_rate": q,
            "steps_per_epoch": steps_per_epoch,
            "planned_steps": planned_steps,
            "sigma": float(sigma),
        }

        print(
            f"  Client {cid}: N={n}, q={q:.8f}, "
            f"steps/epoch={steps_per_epoch}, "
            f"planned_steps={planned_steps:,}, "
            f"sigma={sigma:.6f}"
        )

    print("\nClient-specific DP noise multipliers:")
    for cid in sorted(sigmas):
        print(f"  Client {cid}: sigma={sigmas[cid]:.6f}")

    return sigmas, metadata


# =====================================================================
# FEDPROX + DP-SGD CLIENT TRAINING
# =====================================================================

def add_fedprox_gradient(model, global_reference, mu):
    """
    Add the deterministic FedProx gradient after the DP backward pass.

    The proximal gradient depends only on the round-start global model
    and current model parameters, not on any individual example.
    It is therefore added separately from per-example clipping.
    """
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        reference = global_reference.get(name)
        if reference is None:
            continue

        proximal_gradient = float(mu) * (
            parameter.detach() - reference
        )

        if parameter.grad is None:
            parameter.grad = proximal_gradient.clone()
        else:
            parameter.grad.add_(proximal_gradient)


def train_client_dp(
    global_state, client_id, client_loader, device, class_weights,
    label_smoothing, noise_multiplier, privacy_engine, local_epochs,
    learning_rate, backbone_lr_ratio, weight_decay, fedprox_mu,
    max_grad_norm, target_delta,
):
    """Train one client with Opacus 1.6 hooks-based DP-SGD and FedProx (physical batch size 4, no gradient accumulation).

    The critical integration fix is to invoke Opacus' normal
    private_optimizer.step() rather than calling original_optimizer.step()
    directly. FedProx is injected by wrapping the underlying optimizer step,
    so the sequence is: DP clipping/noise -> FedProx -> AdamW -> accountant.
    """
    local_model, bn_count, relu_count, sd_count, dropout_count = build_dp_model(device, pretrained=False)
    local_model.load_state_dict(global_state, strict=True)
    local_model.train()
    epsilon_before = privacy_engine.get_epsilon(float(target_delta))

    named_parameters = dict(local_model.named_parameters())
    global_reference = {
        name: tensor.to(device, non_blocking=True).detach().clone()
        for name, tensor in global_state.items() if name in named_parameters
    }

    backbone_parameters = [
        p for p in local_model.backbone.parameters()
        if p.requires_grad
    ]
    backbone_ids = {id(p) for p in backbone_parameters}
    head_parameters = [
        p for p in local_model.parameters()
        if p.requires_grad and id(p) not in backbone_ids
    ]

    if not backbone_parameters:
        raise RuntimeError(
            f"Client {client_id}: partial TL produced no trainable backbone parameters."
        )
    if not head_parameters:
        raise RuntimeError(
            f"Client {client_id}: hybrid head has no trainable parameters."
        )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": float(learning_rate) * float(backbone_lr_ratio),
            },
            {
                "params": head_parameters,
                "lr": float(learning_rate),
            },
        ],
        weight_decay=float(weight_decay),
    )
    criterion = build_train_criterion(class_weights, label_smoothing)

    private_model, private_optimizer, private_loader = privacy_engine.make_private(
        module=local_model, optimizer=optimizer, criterion=criterion,
        data_loader=client_loader, noise_multiplier=float(noise_multiplier),
        max_grad_norm=float(max_grad_norm), poisson_sampling=True,
        clipping="flat", grad_sample_mode="hooks"
    )
    # Opacus 1.6.x returns (module, optimizer, data_loader).
    # The training criterion remains the original criterion passed above.
    private_criterion = criterion

    print(f"  Client {client_id}: BN->GN={bn_count}, inplace-ReLU fixed={relu_count}, StochasticDepth disabled={sd_count}, Dropout disabled={dropout_count}")
    print(f"  DP sigma              : {noise_multiplier:.6f}")
    print(f"  DP sample rate        : {private_loader.sample_rate:.8f}")
    print(f"  Fixed FedProx mu      : {fedprox_mu:.6f}")
    print(f"  Task-head LR          : {learning_rate:.2e}")
    print(f"  Backbone LR           : {learning_rate * backbone_lr_ratio:.2e}")
    print("  Physical batch size   : 4")
    print("  Gradient accumulation : 1 (none)")

    # Do not bypass Opacus with original_optimizer.step(). Wrap that call so
    # the normal DPOptimizer.step() remains responsible for DP preparation
    # and privacy accounting.
    original_step = private_optimizer.original_optimizer.step
    prox_state = {"added": False}

    def original_step_with_fedprox(closure=None):
        add_fedprox_gradient(local_model, global_reference, fedprox_mu)
        prox_state["added"] = True
        return original_step() if closure is None else original_step(closure)

    private_optimizer.original_optimizer.step = original_step_with_fedprox

    histories, final_loss, final_accuracy, final_total = [], 0.0, 0.0, 0
    optimizer_steps = 0
    nonzero_grad_observations = 0

    probe_name, probe_before = None, None
    for name, parameter in local_model.named_parameters():
        if parameter.requires_grad and parameter.numel() > 0:
            probe_name = name
            probe_before = parameter.detach().clone()
            break

    for epoch in range(1, local_epochs + 1):
        epoch_loss, epoch_correct, epoch_total, epoch_steps = 0.0, 0, 0, 0
        progress = tqdm(private_loader, desc=f"    Client {client_id} epoch {epoch}/{local_epochs} [DP]", leave=False)

        for images, features, labels in progress:
            if labels.numel() == 0:
                progress.set_postfix(loss="empty", acc=f"{epoch_correct / epoch_total:.4f}" if epoch_total else "0.0000", eps=f"{privacy_engine.get_epsilon(float(target_delta)):.4f}")
                continue

            images = images.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            private_optimizer.zero_grad(set_to_none=True)
            prox_state["added"] = False

            logits = private_model(images, features)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Client {client_id}: non-finite logits.")
            loss = private_criterion(logits, labels)
            loss_check = loss.detach().as_subclass(torch.Tensor)
            if not torch.isfinite(loss_check).all():
                raise FloatingPointError(f"Client {client_id}: non-finite DP loss.")
            loss.backward()

            # Full Opacus step: clipping + noise + wrapped FedProx + AdamW.
            private_optimizer.step()
            optimizer_steps += 1
            epoch_steps += 1
            if not prox_state["added"]:
                raise RuntimeError(f"Client {client_id}: FedProx hook was not executed.")

            nonzero_grad_observations += sum(1 for p in local_model.parameters() if p.grad is not None)
            bs = int(labels.size(0))
            loss_scalar = float(loss_check.item())
            epoch_loss += loss_scalar * bs
            epoch_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            epoch_total += bs
            progress.set_postfix(loss=f"{loss_scalar:.4f}", acc=f"{epoch_correct / epoch_total:.4f}", eps=f"{privacy_engine.get_epsilon(float(target_delta)):.4f}")

        if epoch_total == 0:
            raise RuntimeError(f"Client {client_id} produced only empty Poisson batches in epoch {epoch}.")
        epoch_loss /= epoch_total
        epoch_accuracy = epoch_correct / epoch_total
        epsilon_now = privacy_engine.get_epsilon(float(target_delta))
        print(f"      Client {client_id} local epoch {epoch}: loss={epoch_loss:.4f}, acc={epoch_accuracy:.4f}, steps={epoch_steps}, epsilon={epsilon_now:.6f}")
        histories.append({"epoch": epoch, "loss": epoch_loss, "accuracy": epoch_accuracy, "optimizer_steps": epoch_steps, "epsilon": float(epsilon_now)})
        final_loss, final_accuracy, final_total = epoch_loss, epoch_accuracy, epoch_total

    if optimizer_steps <= 0:
        raise RuntimeError(f"Client {client_id}: zero DP optimizer steps executed.")
    private_optimizer.original_optimizer.step = original_step

    final_epsilon = privacy_engine.get_epsilon(float(target_delta))
    if final_epsilon + 1e-8 < epsilon_before:
        raise RuntimeError(f"Client {client_id}: PrivacyEngine epsilon decreased from {epsilon_before:.6f} to {final_epsilon:.6f}.")

    local_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
    for key, value in local_state.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f"Client {client_id}: non-finite state tensor '{key}'.")

    drift = compute_state_drift(global_state, local_state)
    if drift <= 1e-12:
        probe_change = 0.0
        if probe_name is not None:
            probe_after = dict(local_model.named_parameters())[probe_name].detach()
            probe_change = float(torch.max(torch.abs(probe_after - probe_before)).item())
        raise FloatingPointError(f"Client {client_id}: local model did not change. drift={drift:.6e}, probe_change={probe_change:.6e}, optimizer_steps={optimizer_steps}.")
    if nonzero_grad_observations <= 0:
        raise FloatingPointError(f"Client {client_id}: no gradients were observed.")

    print(f"  Client {client_id} verified local update: drift={drift:.6e}, optimizer_steps={optimizer_steps}")
    del private_model, private_optimizer, private_criterion
    aggressive_cuda_cleanup(f"client {client_id} finished")
    return local_state, final_total, final_loss, final_accuracy, final_epsilon, drift, histories


# =====================================================================
# DP-SAFE SERVER-SIDE SMART AGGREGATION + AUTOMATIC FEDYOGI
# =====================================================================

# These operations consume no additional record-level privacy budget when
# applied only to the DP-protected client model updates received by the server.
V12_DP_FEDYOGI_INITIAL_LR = 2e-4
V12_DP_FEDYOGI_MIN_LR = 5e-5
V12_DP_FEDYOGI_MAX_LR = 8e-4
V12_DP_FEDYOGI_BETA1 = 0.90
V12_DP_FEDYOGI_INITIAL_BETA2 = 0.99
V12_DP_FEDYOGI_MIN_BETA2 = 0.95
V12_DP_FEDYOGI_MAX_BETA2 = 0.999
V12_DP_FEDYOGI_EPSILON = 1e-8
V12_DP_FEDYOGI_BETA2_SMOOTHING = 0.20

V12_DP_CLIENT_WEIGHT_GAMMA = 0.50
V12_DP_CLIENT_ENTROPY_STRENGTH = 0.50
V12_DP_CLIENT_RELIABILITY_STRENGTH = 0.50
V12_DP_CLIENT_DRIFT_STRENGTH = 0.50
V12_DP_CLIENT_WEIGHT_MIN = 0.50
V12_DP_CLIENT_WEIGHT_MAX = 2.00


def _state_rms(values):
    rms_values = []
    for value in values:
        value = value.float()
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                "Non-finite tensor encountered while computing RMS."
            )
        rms_values.append(float(torch.sqrt(torch.mean(value.square())).item()))
    if not rms_values:
        raise ValueError("No floating-point tensors available for RMS calculation.")
    return float(np.sqrt(np.mean(np.square(rms_values))))


def _flatten_state_delta(delta):
    parts = []
    for key, value in delta.items():
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
        if not torch.is_floating_point(global_tensor):
            continue
        current = client_state[key].float()
        base = global_tensor.float()
        if not torch.isfinite(current).all():
            raise FloatingPointError(
                f"Non-finite client state detected in '{key}'."
            )
        d = current - base
        if not torch.isfinite(d).all():
            raise FloatingPointError(
                f"Non-finite client delta detected in '{key}'."
            )
        delta[key] = d
    return delta


def _client_class_entropy(dataset):
    if not hasattr(dataset, "df"):
        raise AttributeError("Client dataset must expose dataset.df.")
    labels = dataset.df["label"].tolist()
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


def compute_smart_client_weights(
    global_state,
    client_states,
    client_sample_counts,
    client_entropies,
    client_drifts,
):
    """
    Two-pass smart aggregation over DP-protected client updates.

    Base:
        N_i ** gamma

    Multipliers:
        entropy diversity
        reliability from cosine similarity to provisional aggregate
        drift relative to median client drift

    Final multiplier is clipped to a conservative range and normalized.
    """
    n_clients = len(client_states)
    if n_clients == 0:
        raise ValueError("No client states supplied.")
    if not (
        len(client_sample_counts) == n_clients
        and len(client_entropies) == n_clients
        and len(client_drifts) == n_clients
    ):
        raise ValueError("Smart-aggregation metadata lengths do not match.")

    gamma = V12_DP_CLIENT_WEIGHT_GAMMA
    entropy_strength = V12_DP_CLIENT_ENTROPY_STRENGTH
    reliability_strength = V12_DP_CLIENT_RELIABILITY_STRENGTH
    drift_strength = V12_DP_CLIENT_DRIFT_STRENGTH

    base_weights = np.asarray(
        [max(float(n), 1.0) ** gamma for n in client_sample_counts],
        dtype=np.float64,
    )

    provisional = base_weights * np.asarray(
        [
            1.0 + entropy_strength * (float(h) - 0.5)
            for h in client_entropies
        ],
        dtype=np.float64,
    )
    provisional = np.maximum(provisional, 1e-12)
    provisional /= provisional.sum()

    client_vectors = [
        _flatten_state_delta(_client_delta(global_state, state))
        for state in client_states
    ]
    provisional_vector = sum(
        float(w) * vector
        for w, vector in zip(provisional, client_vectors)
    )

    similarities = np.asarray(
        [
            _cosine_similarity(vector, provisional_vector)
            for vector in client_vectors
        ],
        dtype=np.float64,
    )

    raw_reliability = 0.75 + 0.25 * similarities
    reliability = (
        1.0 + reliability_strength * (raw_reliability - 1.0)
    )

    finite_drifts = np.asarray(
        [max(float(d), 1e-12) for d in client_drifts],
        dtype=np.float64,
    )
    median_drift = float(np.median(finite_drifts))
    drift_ratio = finite_drifts / max(median_drift, 1e-12)
    raw_drift = np.sqrt(1.0 / np.maximum(drift_ratio, 1e-12))
    raw_drift = np.clip(raw_drift, 0.75, 1.25)
    drift_factor = 1.0 + drift_strength * (raw_drift - 1.0)

    multipliers = np.asarray(
        [
            float(
                np.clip(
                    (1.0 + entropy_strength * (float(h) - 0.5))
                    * float(r)
                    * float(d),
                    V12_DP_CLIENT_WEIGHT_MIN,
                    V12_DP_CLIENT_WEIGHT_MAX,
                )
            )
            for h, r, d in zip(client_entropies, reliability, drift_factor)
        ],
        dtype=np.float64,
    )

    final_weights = base_weights * multipliers
    final_weights = np.maximum(final_weights, 1e-12)
    final_weights /= final_weights.sum()

    diagnostics = []
    for i in range(n_clients):
        diagnostics.append(
            {
                "base_sqrt_size": float(base_weights[i]),
                "entropy": float(client_entropies[i]),
                "cosine_reliability": float(similarities[i]),
                "drift": float(client_drifts[i]),
                "final_multiplier": float(multipliers[i]),
                "normalized_weight": float(final_weights[i]),
            }
        )

    return final_weights.tolist(), diagnostics


class AutomaticFedYogiDP:
    """
    Automatic FedYogi applied only to DP-protected client updates.

    The server sees no raw client examples. The server optimizer operates
    exclusively on the DP-protected model updates received from clients.
    """

    def __init__(self):
        self.server_lr = V12_DP_FEDYOGI_INITIAL_LR
        self.min_lr = V12_DP_FEDYOGI_MIN_LR
        self.max_lr = V12_DP_FEDYOGI_MAX_LR

        self.beta1 = V12_DP_FEDYOGI_BETA1
        self.beta2 = V12_DP_FEDYOGI_INITIAL_BETA2
        self.min_beta2 = V12_DP_FEDYOGI_MIN_BETA2
        self.max_beta2 = V12_DP_FEDYOGI_MAX_BETA2
        self.epsilon = V12_DP_FEDYOGI_EPSILON
        self.beta2_smoothing = V12_DP_FEDYOGI_BETA2_SMOOTHING

        self.step_count = 0
        self.first_moment = {}
        self.second_moment = {}
        self.previous_delta_vector = None
        self.update_rms_ema = None

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
            raise FloatingPointError("FedYogi RMS received NaN/Inf.")
        return float(torch.sqrt(torch.mean(tensor.square())).item())

    def _adapt_hyperparameters(self, current_update_rms, current_vector):
        if self.previous_delta_vector is None:
            volatility = 0.0
            magnitude_ratio = 1.0
            self.update_rms_ema = float(current_update_rms)
        else:
            cosine = _cosine_similarity(
                current_vector,
                self.previous_delta_vector,
            )
            volatility = float(np.clip(1.0 - cosine, 0.0, 2.0))
            magnitude_ratio = float(
                current_update_rms / max(self.update_rms_ema, 1e-12)
            )
            self.update_rms_ema = (
                0.80 * self.update_rms_ema
                + 0.20 * current_update_rms
            )

        size_signal = float(
            np.clip(np.log(max(magnitude_ratio, 1e-6)), -3.0, 3.0)
        )
        volatility_signal = float(np.clip(volatility, 0.0, 2.0))

        lr_factor = float(
            np.exp(
                -0.06 * size_signal
                -0.12 * volatility_signal
            )
        )
        lr_factor = float(np.clip(lr_factor, 0.85, 1.15))
        self.server_lr = float(
            np.clip(
                self.server_lr * lr_factor,
                self.min_lr,
                self.max_lr,
            )
        )

        target_beta2 = self.max_beta2 - (
            self.max_beta2 - self.min_beta2
        ) * (volatility_signal / 2.0)
        target_beta2 = float(
            np.clip(target_beta2, self.min_beta2, self.max_beta2)
        )
        self.beta2 = float(
            np.clip(
                (1.0 - self.beta2_smoothing) * self.beta2
                + self.beta2_smoothing * target_beta2,
                self.min_beta2,
                self.max_beta2,
            )
        )

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
        weights, diagnostics = compute_smart_client_weights(
            global_state,
            client_states,
            client_sample_counts,
            client_entropies,
            client_drifts,
        )

        delta = {}
        for key, global_tensor in global_state.items():
            if not torch.is_floating_point(global_tensor):
                continue

            base = global_tensor.float()
            result = torch.zeros_like(base, dtype=torch.float32)

            for state, weight in zip(client_states, weights):
                current = state[key].float()
                if not torch.isfinite(current).all():
                    raise FloatingPointError(
                        f"Non-finite DP client state in '{key}'."
                    )
                result.add_(
                    current - base,
                    alpha=float(weight),
                )

            if not torch.isfinite(result).all():
                raise FloatingPointError(
                    f"Non-finite smart aggregate delta in '{key}'."
                )
            delta[key] = result

        current_update_rms = _state_rms(delta.values())
        current_vector = _flatten_state_delta(delta)

        magnitude_ratio, volatility, lr_factor = (
            self._adapt_hyperparameters(
                current_update_rms,
                current_vector,
            )
        )

        candidate_step = self.step_count + 1
        bc1 = 1.0 - self.beta1 ** candidate_step
        bc2 = 1.0 - self.beta2 ** candidate_step

        candidate_m = {}
        candidate_v = {}
        candidate_directions = {}
        direction_rms_values = []

        for key, current_delta in delta.items():
            old_m = self.first_moment.get(
                key,
                torch.zeros_like(current_delta),
            ).float()
            old_v = self.second_moment.get(
                key,
                torch.zeros_like(current_delta),
            ).float()

            new_m = (
                self.beta1 * old_m
                + (1.0 - self.beta1) * current_delta
            )
            delta_sq = current_delta.square()
            new_v = old_v - (
                1.0 - self.beta2
            ) * torch.sign(old_v - delta_sq) * delta_sq
            new_v = torch.clamp(new_v, min=0.0)

            if not torch.isfinite(new_m).all() or not torch.isfinite(new_v).all():
                raise FloatingPointError("Non-finite FedYogi moment.")

            m_hat = new_m / bc1
            v_hat = new_v / bc2
            direction = m_hat / (
                torch.sqrt(v_hat) + self.epsilon
            )

            if not torch.isfinite(direction).all():
                raise FloatingPointError("Non-finite FedYogi direction.")

            candidate_m[key] = new_m
            candidate_v[key] = new_v
            direction_rms_values.append(self._rms(direction))

        direction_rms = float(
            np.sqrt(np.mean(np.square(direction_rms_values)))
        )
        if not np.isfinite(direction_rms) or direction_rms <= 0:
            raise FloatingPointError(
                f"Invalid FedYogi direction RMS: {direction_rms}"
            )

        candidate_state = {}
        applied_rms_values = []

        for key, global_tensor in global_state.items():
            if key not in candidate_m:
                candidate_state[key] = global_tensor.clone()
                continue

            m_hat = candidate_m[key] / bc1
            v_hat = candidate_v[key] / bc2
            direction = m_hat / (
                torch.sqrt(v_hat) + self.epsilon
            )

            candidate_tensor = (
                global_tensor.float()
                + self.server_lr * direction
            )

            if not torch.isfinite(candidate_tensor).all():
                raise FloatingPointError(
                    f"FedYogi produced NaN/Inf in '{key}'."
                )

            applied_rms_values.append(
                self._rms(candidate_tensor - global_tensor.float())
            )
            candidate_state[key] = candidate_tensor.to(
                dtype=global_tensor.dtype
            )

        applied_rms = float(
            np.sqrt(np.mean(np.square(applied_rms_values)))
        )

        self.first_moment = {
            k: v.detach().cpu().clone()
            for k, v in candidate_m.items()
        }
        self.second_moment = {
            k: v.detach().cpu().clone()
            for k, v in candidate_v.items()
        }
        self.step_count = candidate_step
        self.last_update_rms = current_update_rms
        self.last_direction_rms = direction_rms
        self.last_applied_update_rms = applied_rms
        self.last_client_weights = list(weights)
        self.last_client_weight_diagnostics = diagnostics

        return candidate_state, weights, {
            "magnitude_ratio": magnitude_ratio,
            "volatility": volatility,
            "lr_factor": lr_factor,
        }


    def load_state_dict(self, state):
        self.step_count = int(state["step_count"])
        self.server_lr = float(state["server_lr"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])

        self.first_moment = {
            k: v.detach().cpu().clone()
            for k, v in state["first_moment"].items()
        }
        self.second_moment = {
            k: v.detach().cpu().clone()
            for k, v in state["second_moment"].items()
        }

        previous = state.get("previous_delta_vector")
        self.previous_delta_vector = (
            previous.detach().cpu().clone()
            if previous is not None else None
        )

        self.update_rms_ema = state.get("update_rms_ema")
        self.last_update_rms = float(state.get("last_update_rms", 0.0))
        self.last_direction_rms = float(state.get("last_direction_rms", 0.0))
        self.last_applied_update_rms = float(
            state.get("last_applied_update_rms", 0.0)
        )
        self.last_volatility = float(state.get("last_volatility", 0.0))
        self.last_beta2_target = float(
            state.get("last_beta2_target", self.beta2)
        )
        self.last_client_weights = list(
            state.get("last_client_weights", [])
        )
        self.last_client_weight_diagnostics = list(
            state.get("last_client_weight_diagnostics", [])
        )

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "server_lr": self.server_lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "first_moment": {
                k: v.clone() for k, v in self.first_moment.items()
            },
            "second_moment": {
                k: v.clone() for k, v in self.second_moment.items()
            },
            "previous_delta_vector": self.previous_delta_vector,
            "update_rms_ema": self.update_rms_ema,
            "last_update_rms": self.last_update_rms,
            "last_direction_rms": self.last_direction_rms,
            "last_applied_update_rms": self.last_applied_update_rms,
            "last_volatility": self.last_volatility,
            "last_beta2_target": self.last_beta2_target,
            "last_client_weights": list(self.last_client_weights),
            "last_client_weight_diagnostics": list(
                self.last_client_weight_diagnostics
            ),
        }


# =====================================================================
# STATE / FEDAVG
# =====================================================================

def validate_state_dict(new_state, reference_state):
    if set(new_state) != set(reference_state):
        raise RuntimeError("Global state-dict keys changed during aggregation.")
    for key, ref in reference_state.items():
        value = new_state[key]
        if value.shape != ref.shape or value.dtype != ref.dtype:
            raise RuntimeError(f"Invalid aggregated state tensor '{key}'.")
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite aggregated tensor '{key}'.")


def compute_state_drift(global_state, client_state):
    sq_diff = 0.0
    sq_global = 0.0

    for key, base in global_state.items():
        if not torch.is_floating_point(base):
            continue
        current = client_state[key].float()
        base_f = base.float()
        sq_diff += float(torch.sum((current - base_f) ** 2).item())
        sq_global += float(torch.sum(base_f ** 2).item())

    return float(np.sqrt(sq_diff) / (np.sqrt(sq_global) + 1e-12))


def fedavg(global_state, client_states, client_sample_counts):
    if not client_states:
        raise ValueError("No client states supplied to FedAvg.")

    weights = np.asarray(client_sample_counts, dtype=np.float64)
    if np.any(weights <= 0):
        raise ValueError("FedAvg client sample counts must be positive.")
    weights /= weights.sum()

    new_state = {}

    for key, base in global_state.items():
        if torch.is_floating_point(base):
            result = torch.zeros_like(base, dtype=torch.float32)
            for state, weight in zip(client_states, weights):
                result.add_(state[key].float(), alpha=float(weight))
            new_state[key] = result.to(dtype=base.dtype)
        else:
            # No BatchNorm buffers are expected after GN conversion.
            # For any non-floating buffer, retain the global value.
            new_state[key] = base.clone()

    return new_state, weights.tolist()


# =====================================================================
# EVALUATION
# =====================================================================

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, features, labels in tqdm(
        loader, desc="  evaluate", leave=False
    ):
        images = images.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, features)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * int(labels.size(0))
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, acc, macro_f1, balanced_acc, all_preds, all_labels


def per_class_f1(labels, predictions):
    scores = f1_score(
        labels,
        predictions,
        labels=list(range(config.NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    return {
        cls: float(score)
        for cls, score in zip(config.CLASS_LIST, scores)
    }


def save_confusion_matrix(labels, predictions, path):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(config.NUM_CLASSES)),
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm)
    indices = list(range(config.NUM_CLASSES))
    ax.set_xticks(indices)
    ax.set_yticks(indices)
    ax.set_xticklabels(config.CLASS_LIST)
    ax.set_yticklabels(config.CLASS_LIST)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Federated MaxViT-Tiny V2 - DP-SGD Test Confusion Matrix")

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# CHECKPOINT / RESUME
# =====================================================================

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_accountant_state(privacy_engine):
    """
    Persist the actual Opacus accountant history when supported.
    RDPAccountant composes privacy from its recorded mechanisms/steps.
    """
    accountant = getattr(privacy_engine, "accountant", None)
    if accountant is None:
        raise RuntimeError("PrivacyEngine has no accountant to checkpoint.")

    if not hasattr(accountant, "state_dict"):
        raise RuntimeError(
            "Installed Opacus accountant does not expose state_dict(); "
            "cannot create a resume-safe DP checkpoint."
        )

    return accountant.state_dict()


def restore_accountant_state(privacy_engine, state):
    accountant = getattr(privacy_engine, "accountant", None)
    if accountant is None:
        raise RuntimeError("PrivacyEngine has no accountant to restore.")

    if not hasattr(accountant, "load_state_dict"):
        raise RuntimeError(
            "Installed Opacus accountant does not expose load_state_dict(); "
            "cannot restore the DP accountant exactly."
        )

    accountant.load_state_dict(state)


def capture_all_privacy_states(privacy_engines):
    return {
        int(cid): capture_accountant_state(engine)
        for cid, engine in privacy_engines.items()
    }


def restore_all_privacy_states(privacy_engines, states):
    if not states:
        raise RuntimeError("Checkpoint contains no privacy accountant states.")

    for cid, engine in privacy_engines.items():
        key = int(cid)
        if key not in states and str(key) not in states:
            raise RuntimeError(
                f"Checkpoint is missing accountant state for client {cid}."
            )
        state = states[key] if key in states else states[str(key)]
        restore_accountant_state(engine, state)


# ---------------------------------------------------------------------
# ROBUST CHECKPOINT WRITER
# ---------------------------------------------------------------------
# The project is under OneDrive on Windows. Large PyTorch ZIP checkpoints
# can fail when they are written directly into a synchronized directory:
#
#   ios_base::badbit set: iostream stream error
#   unexpected pos ... vs ...
#
# Serialize completely in the local Windows TEMP directory first, flush it
# to disk, and only then move the completed file into the project directory.
def _robust_torch_save(checkpoint, path):
    import tempfile

    path = os.path.abspath(str(path))
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)

    temp_root = tempfile.gettempdir()
    os.makedirs(temp_root, exist_ok=True)

    tmp_path = os.path.join(
        temp_root,
        f"13C_checkpoint_{os.getpid()}_{time.time_ns()}_"
        f"{os.path.basename(path)}.tmp",
    )

    try:
        print(f"  Serializing checkpoint outside OneDrive: {tmp_path}")

        with open(tmp_path, "wb") as fh:
            torch.save(checkpoint, fh)
            fh.flush()
            os.fsync(fh.fileno())

        if not os.path.isfile(tmp_path):
            raise IOError(f"Temporary checkpoint was not created: {tmp_path}")

        tmp_size = os.path.getsize(tmp_path)
        if tmp_size <= 0:
            raise IOError(f"Temporary checkpoint is empty: {tmp_path}")

        print(
            f"  Checkpoint serialization complete: "
            f"{tmp_size / (1024 ** 2):.2f} MB"
        )

        # TEMP and the project are normally on the same Windows volume.
        # os.replace moves the completed file instead of writing the large
        # checkpoint progressively into the OneDrive directory.
        os.replace(tmp_path, path)

        if not os.path.isfile(path):
            raise IOError(f"Checkpoint destination was not created: {path}")

        final_size = os.path.getsize(path)
        if final_size != tmp_size:
            raise IOError(
                f"Checkpoint size changed during commit: "
                f"temporary={tmp_size}, destination={final_size}"
            )

        print(
            f"  Checkpoint committed: {path} "
            f"({final_size / (1024 ** 2):.2f} MB)"
        )

    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def save_resume_checkpoint(
    path,
    model,
    global_state,
    round_number,
    best_val_f1,
    best_val_acc,
    best_val_loss,
    best_round,
    server_optimizer,
    privacy_engines,
    client_sigmas,
    sigma_metadata,
    epsilon_history,
    log_rows,
    config_snapshot,
):
    checkpoint = {
        "checkpoint_type": "13C_FULL_RESUME",
        "model_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        },
        "global_state": {
            k: v.detach().cpu().clone()
            for k, v in global_state.items()
        },
        "round": int(round_number),
        "next_round": int(round_number) + 1,
        "best_val_f1": float(best_val_f1),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_round": int(best_round),
        "server_optimizer_state": server_optimizer.state_dict(),
        "privacy_accountant_states": capture_all_privacy_states(privacy_engines),
        "client_sigmas": {
            int(cid): float(sigma)
            for cid, sigma in client_sigmas.items()
        },
        "sigma_metadata": sigma_metadata,
        "epsilon_history": epsilon_history,
        "log_rows": log_rows,
        "rng_state": capture_rng_state(),
        "config": config_snapshot,
        "experiment": config_snapshot.get("experiment", "13C"),
    }

    _robust_torch_save(checkpoint, path)


def save_checkpoint(
    path,
    model,
    round_number,
    val_loss,
    val_acc,
    val_macro_f1,
    best_val_f1,
    config_snapshot,
    privacy_snapshot=None,
):
    """Save the best-model checkpoint used for final test evaluation."""
    checkpoint = {
        "checkpoint_type": "13C_BEST_MODEL",
        "model_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        },
        "round": int(round_number),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_macro_f1": float(val_macro_f1),
        "best_val_f1": float(best_val_f1),
        "class_list": list(config.CLASS_LIST),
        "num_classes": int(config.NUM_CLASSES),
        "handcrafted_feature_dim": int(config.HANDCRAFTED_FEATURE_DIM),
        "random_seed": int(config.RANDOM_SEED),
        "config": config_snapshot,
        "privacy": privacy_snapshot or {},
    }

    _robust_torch_save(checkpoint, path)


def load_resume_checkpoint(
    path,
    global_model,
    privacy_engines,
    server_optimizer,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "global_state",
        "round",
        "server_optimizer_state",
        "privacy_accountant_states",
        "rng_state",
    }
    missing = sorted(required - set(checkpoint.keys()))
    if missing:
        raise RuntimeError(
            "Checkpoint is not resume-safe. Missing keys: "
            + ", ".join(missing)
        )

    saved_config = checkpoint.get("config", {})
    saved_partition_dir = saved_config.get("partition_dir")
    saved_client_sizes = tuple(saved_config.get("client_sizes", []))
    expected_partition_dir = os.path.normcase(os.path.abspath(ALPHA100_PARTITION_DIR))
    if saved_partition_dir is None or os.path.normcase(os.path.abspath(saved_partition_dir)) != expected_partition_dir:
        raise RuntimeError(
            "13C REFUSED TO RESUME: checkpoint partition directory does not "
            "match the locked alpha_100.0 directory. "
            f"Expected={ALPHA100_PARTITION_DIR}, found={saved_partition_dir}"
        )
    if saved_client_sizes != EXPECTED_ALPHA100_CLIENT_SIZES:
        raise RuntimeError(
            "13C REFUSED TO RESUME: checkpoint client sizes do not match the "
            f"verified alpha_100.0 sizes. Expected={EXPECTED_ALPHA100_CLIENT_SIZES}, "
            f"found={saved_client_sizes}"
        )

    global_state = {
        k: v.detach().cpu().clone()
        for k, v in checkpoint["global_state"].items()
    }
    global_model.load_state_dict(global_state, strict=True)

    server_state = checkpoint["server_optimizer_state"]
    server_optimizer.load_state_dict(server_state)

    restore_all_privacy_states(
        privacy_engines,
        checkpoint["privacy_accountant_states"],
    )
    restore_rng_state(checkpoint["rng_state"])

    start_round = int(checkpoint["round"]) + 1

    print("\n" + "=" * 78)
    print("RESUMED FROM FULL ROUND CHECKPOINT")
    print("=" * 78)
    print(f"Completed round           : {checkpoint['round']}")
    print(f"Next round                : {start_round}")
    print(f"Best round                : {checkpoint.get('best_round', 0)}")
    print(
        f"Best validation Macro-F1  : "
        f"{checkpoint.get('best_val_f1', -1.0):.6f}"
    )
    print("=" * 78)

    return (
        global_state,
        start_round,
        float(checkpoint.get("best_val_f1", -1.0)),
        float(checkpoint.get("best_val_acc", 0.0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
        int(checkpoint.get("best_round", 0)),
        checkpoint.get("epsilon_history", {}),
        checkpoint.get("log_rows", []),
        checkpoint.get("client_sigmas"),
        checkpoint.get("sigma_metadata"),
    )


# =====================================================================
# MAIN
# =====================================================================


def plot_training_curves(log_df, path):
    if log_df.empty:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    axes[0].plot(log_df["round"], log_df["val_loss"], marker="o")
    axes[0].set_xlabel("Federated Round")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("DP Federated Validation Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(log_df["round"], log_df["val_acc"], marker="o", label="Validation Accuracy")
    axes[1].plot(log_df["round"], log_df["val_macro_f1"], marker="s", label="Validation Macro-F1")
    axes[1].set_xlabel("Federated Round")
    axes[1].set_ylabel("Score")
    axes[1].set_title("DP Federated Validation Metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(log_df["round"], log_df["mean_client_loss"], marker="o", label="Mean Client Loss")
    axes[2].plot(log_df["round"], log_df["mean_client_acc"], marker="s", label="Mean Client Accuracy")
    axes[2].set_xlabel("Federated Round")
    axes[2].set_ylabel("Value")
    axes[2].set_title("DP Client Training")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    start_time = time.perf_counter()

    print("=" * 78)
    print("PHASE 4 - FINAL DP FEDERATED EXPERIMENT - EPSILON 3")
    print("MAXVIT-TINY V2 + PARTIAL TRANSFER LEARNING + GROUPNORM + DP-SGD + FEDPROX + FEDYOGI + SMART AGG")
    print("=" * 78)

    device = get_device()
    if str(opacus.__version__) != "1.6.0":
        raise RuntimeError(
            f"This experiment is locked to Opacus 1.6.0, but found "
            f"Opacus {opacus.__version__}."
        )

    seed = int(getattr(config, "RANDOM_SEED", 42))
    set_random_seeds(seed)

    # -----------------------------------------------------------------
    # FINAL EXPERIMENT SETTINGS
    # -----------------------------------------------------------------
    num_rounds = 10
    local_epochs = 3
    batch_size = 4
    clients_per_round = 6

    target_epsilon = 3.0
    target_delta = 1e-5
    max_grad_norm = 0.75

    # Discriminative learning rate retained from V12.1.
    head_lr = 1e-4
    backbone_lr_ratio = 0.10
    backbone_lr = head_lr * backbone_lr_ratio
    weight_decay = float(
        getattr(
            config,
            "FED_WEIGHT_DECAY",
            getattr(config, "WEIGHT_DECAY", 1e-4),
        )
    )

    # Fixed FedProx: no private-data-driven adaptive mu.
    fedprox_mu = 0.005

    label_smoothing = float(
        getattr(
            config,
            "FED_LABEL_SMOOTHING",
            getattr(config, "LABEL_SMOOTHING", 0.02),
        )
    )

    # -----------------------------------------------------------------
    # SERVER: FedYogi + smart aggregation
    # -----------------------------------------------------------------
    # These operate only on the DP-protected client model states.
    server_optimizer = AutomaticFedYogiDP()

    # -----------------------------------------------------------------
    # CLIENT IDs
    # -----------------------------------------------------------------
    client_ids = get_client_ids()

    if len(client_ids) != clients_per_round:
        print(
            f"\nConfigured clients={len(client_ids)}; "
            f"all {len(client_ids)} clients will participate each round."
        )
        clients_per_round = len(client_ids)

    if clients_per_round != len(client_ids):
        raise RuntimeError(
            "This final experiment requires all clients to participate every round."
        )

    # HARD STOP before model training: verify the exact pre-generated
    # alpha_100.0 partition and all six expected client sizes.
    validate_alpha100_partition(client_ids)

    print("\n" + "=" * 78)
    print("FINAL DP PIPELINE")
    print("=" * 78)
    print("Partition                  : HAM10000 lesion-level Dirichlet alpha=100.0")
    print("Architecture               : MaxViT-Tiny V2 hybrid + partial transfer learning")
    print("Initialization             : ImageNet pretrained; early backbone stages frozen")
    print("Transfer learning          : final MaxViT stage + hybrid head trainable")
    print("Normalization              : GroupNorm task heads + MaxViT LayerNorm")
    print("Dropout                    : disabled for Opacus compatibility")
    print("StochasticDepth            : disabled for Opacus compatibility")
    print("Image augmentation         : existing on-the-fly training augmentation")
    print("Loss                       : fixed global class-balanced CE")
    print(f"Label smoothing            : {label_smoothing:.4f}")
    print("Sampling                   : Opacus Poisson")
    print("DP mechanism               : Opacus 1.6 DP-SGD + Ghost clipping")
    print(f"Target epsilon             : {target_epsilon:.4f}")
    print(f"Target delta               : {target_delta:.2e}")
    print(f"Max per-sample grad norm   : {max_grad_norm:.4f}")
    print("Local-update verification  : enabled")
    print("Noise                      : client-specific RDP-calibrated")
    print(f"Federated rounds           : {num_rounds}")
    print(f"Local epochs               : {local_epochs}")
    print(f"Physical batch size        : {batch_size}")
    print(f"Task-head LR               : {head_lr:.2e}")
    print(f"Backbone LR                : {backbone_lr:.2e}")
    print(f"Weight decay               : {weight_decay:.2e}")
    print(f"FedProx                    : fixed mu={fedprox_mu:.4f}")
    print("Server optimizer           : Conservative Automatic FedYogi")
    print("Client aggregation         : sqrt(N) + entropy + reliability + drift")
    print("Client adaptive LR         : disabled")
    print("Adaptive FedProx           : disabled")
    print("WeightedRandomSampler      : disabled")
    print("2x replacement sampling    : disabled")
    print("AMP                        : disabled")
    print("Balanced Accuracy          : validation + final test")
    print("Checkpointing              : full state saved after every round")
    print("Resume                     : automatic from latest completed round")
    print("RNG checkpoint             : Python + NumPy + Torch + CUDA")
    print("Server state checkpoint   : FedYogi moments + adaptive state")
    print("DP accountant checkpoint  : persistent per-client RDP state")
    print("CUDA cleanup              : aggressive between clients/rounds")
    print("Test                       : held-out test used once at the end")
    print("=" * 78)

    model_dir = os.path.join(config.OUTPUT_DIR, "models")
    log_dir = os.path.join(config.OUTPUT_DIR, "training_logs")
    plots_dir = os.path.join(config.OUTPUT_DIR, "plots")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    experiment_tag = (
        "13C_FINAL_dp_eps3_hooks_alpha100_verified_client_sigma_"
        "classbalanced_fedprox005_fedyogi_smartagg_gn_partialTL_3epochs_10rounds_batch4"
    )

    best_model_path = os.path.join(
        model_dir,
        f"federated_maxvit_{experiment_tag}_global_best.pt",
    )
    final_model_path = os.path.join(
        model_dir,
        f"federated_maxvit_{experiment_tag}_global_final.pt",
    )

    resume_checkpoint_path = os.path.join(
        model_dir,
        f"federated_maxvit_{experiment_tag}_resume_latest.pt",
    )
    log_path = os.path.join(
        log_dir,
        f"federated_maxvit_{experiment_tag}_training_log.csv",
    )
    report_path = os.path.join(
        log_dir,
        f"federated_maxvit_{experiment_tag}_test_classification_report.txt",
    )
    per_class_path = os.path.join(
        log_dir,
        f"federated_maxvit_{experiment_tag}_per_class_f1.csv",
    )
    curves_path = os.path.join(
        plots_dir,
        f"federated_maxvit_{experiment_tag}_training_curves.png",
    )
    confusion_path = os.path.join(
        plots_dir,
        f"federated_maxvit_{experiment_tag}_test_confusion_matrix.png",
    )

    # Set to True to continue from the most recent completed-round checkpoint.
    # A checkpoint is written atomically after every successfully completed
    # federated round, so a native CUDA crash during a round leaves the prior
    # round checkpoint intact.
    RESUME_IF_AVAILABLE = True

    # -----------------------------------------------------------------
    # GLOBAL MODEL
    # -----------------------------------------------------------------
    global_model, bn_count, relu_count, sd_count, dropout_count = (
        build_dp_model(device, pretrained=True)
    )

    total_params = sum(p.numel() for p in global_model.parameters())

    print("\nGlobal model:")
    print(f"  Parameters                 : {total_params:,}")
    print(f"  BN -> GN conversions       : {bn_count}")
    print(f"  In-place ReLU conversions  : {relu_count}")
    print(f"  StochasticDepth disabled   : {sd_count}")
    print(f"  Dropout disabled           : {dropout_count}")

    # Explicit final validation after all conversions.
    validator_errors = ModuleValidator.validate(global_model, strict=False)
    if validator_errors:
        raise RuntimeError(
            "Final global model is not Opacus compatible:\n"
            + "\n".join(str(e) for e in validator_errors[:20])
        )
    print("  ✓ Opacus ModuleValidator: 0 compatibility errors")

    val_loader = build_eval_loader("val")
    test_loader = build_eval_loader("test")
    eval_criterion = build_eval_criterion(label_smoothing)

    global_state = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    # -----------------------------------------------------------------
    # CLIENT DATASETS / SIGMA CALIBRATION
    # -----------------------------------------------------------------
    client_datasets = {}
    calibration_loaders = {}

    print("\nTrue client dataset sizes:")
    for cid in client_ids:
        loader = build_standard_client_loader(cid, batch_size)
        client_datasets[cid] = loader.dataset
        calibration_loaders[cid] = loader
        print(f"  Client {cid}: {len(loader.dataset)}")

    total_client_samples = sum(len(client_datasets[cid]) for cid in client_ids)
    if len(client_ids) != 6:
        raise RuntimeError(
            f"This final experiment expects 6 clients, found {len(client_ids)}."
        )
    if total_client_samples != 8112:
        raise RuntimeError(
            "The loaded alpha=100.0 training partition does not contain the "
            f"expected 8112 lesion-level training samples; found {total_client_samples}. "
            "Do not train until the alpha=100.0 partition is loaded correctly."
        )
    print(f"  Verified total client samples: {total_client_samples}")

    # Report the actual client distributions so the generated partition can
    # be verified rather than trusting the alpha label alone.
    print("\nClient class-distribution check:")
    for cid in client_ids:
        ds = client_datasets[cid]
        if hasattr(ds, "df") and "label" in ds.df.columns:
            counts = ds.df["label"].value_counts().to_dict()
            print(f"  Client {cid}: {counts}")

    # Compute class-balanced weights ONCE from the complete training
    # partition. They remain fixed for the entire DP experiment.
    # They are never recomputed from private minibatches or local accuracy.
    FIXED_CLASS_WEIGHTS, global_class_counts, label_smoothing = (
        compute_global_class_weights(client_datasets, device)
    )

    # Do not use the private client label distribution to dynamically alter
    # the DP training loss. The class weights above remain fixed constants.
    print("\nFixed class-balanced CE weights:")
    for cls, weight in zip(
        config.CLASS_LIST,
        FIXED_CLASS_WEIGHTS.detach().cpu().tolist(),
    ):
        print(f"  {cls:6s}: {weight:.6f}")

    client_sigmas, sigma_metadata = calibrate_client_sigmas(
        calibration_loaders,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
    )

    # -----------------------------------------------------------------
    # ONE PERSISTENT PRIVACY ACCOUNTANT PER CLIENT
    # -----------------------------------------------------------------
    privacy_engines = {
        cid: PrivacyEngine(accountant="rdp")
        for cid in client_ids
    }

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_round = 0

    log_rows = []
    epsilon_history = {cid: [] for cid in client_ids}
    round_number = 0
    start_round = 1

    # -----------------------------------------------------------------
    # RESUME FROM THE MOST RECENT COMPLETED ROUND, IF AVAILABLE
    # -----------------------------------------------------------------
    if RESUME_IF_AVAILABLE and os.path.exists(resume_checkpoint_path):
        (
            global_state,
            start_round,
            best_val_f1,
            best_val_acc,
            best_val_loss,
            best_round,
            saved_epsilon_history,
            saved_log_rows,
            saved_sigmas,
            saved_sigma_metadata,
        ) = load_resume_checkpoint(
            resume_checkpoint_path,
            global_model,
            privacy_engines,
            server_optimizer,
        )

        if saved_sigmas is not None:
            client_sigmas = {
                int(cid): float(sigma)
                for cid, sigma in saved_sigmas.items()
            }
        if saved_sigma_metadata is not None:
            sigma_metadata = saved_sigma_metadata

        epsilon_history = {
            int(cid): list(values)
            for cid, values in saved_epsilon_history.items()
        }
        log_rows = list(saved_log_rows)

        if start_round > num_rounds:
            print("All configured rounds are already complete.")
            start_round = num_rounds + 1

    # -----------------------------------------------------------------
    # FEDERATED TRAINING
    # -----------------------------------------------------------------
    for round_number in range(start_round, num_rounds + 1):
        round_start = time.perf_counter()

        print("\n" + "=" * 78)
        print(f"FEDERATED DP ROUND {round_number}/{num_rounds}")
        print("=" * 78)
        print(f"Clients: {client_ids}")

        client_states = []
        client_sample_counts = []
        client_entropies = []
        client_drifts = []
        client_losses = []
        client_accuracies = []
        client_epsilons = []
        client_histories = {}

        for cid in client_ids:
            print("\n" + "-" * 70)
            print(f"CLIENT {cid}")
            print("-" * 70)

            client_loader = build_standard_client_loader(cid, batch_size, dataset=client_datasets[cid])
            original_n = len(client_loader.dataset)

            if original_n <= 0:
                raise RuntimeError(f"Client {cid} contains zero samples.")

            entropy = _client_class_entropy(client_loader.dataset)

            print(f"True samples              : {original_n}")
            print(f"Class-distribution entropy: {entropy:.6f}")
            print(f"Sigma                     : {client_sigmas[cid]:.6f}")
            print(f"Sampling                  : Poisson")
            print(f"Task-head LR              : {head_lr:.2e}")
            print(f"Backbone LR               : {backbone_lr:.2e}")
            print(f"FedProx mu                : {fedprox_mu:.4f}")

            (
                local_state,
                final_total,
                client_loss,
                client_acc,
                client_eps,
                drift,
                history,
            ) = train_client_dp(
                global_state=global_state,
                client_id=cid,
                client_loader=client_loader,
                device=device,
                class_weights=FIXED_CLASS_WEIGHTS,
                label_smoothing=label_smoothing,
                noise_multiplier=client_sigmas[cid],
                privacy_engine=privacy_engines[cid],
                local_epochs=local_epochs,
                learning_rate=head_lr,
                backbone_lr_ratio=backbone_lr_ratio,
                weight_decay=weight_decay,
                fedprox_mu=fedprox_mu,
                max_grad_norm=max_grad_norm,
                target_delta=target_delta,
            )

            if client_eps > target_epsilon + 0.01:
                raise RuntimeError(
                    f"Client {cid} exceeded target epsilon: "
                    f"{client_eps:.6f} > {target_epsilon:.6f}"
                )

            client_states.append(local_state)
            client_sample_counts.append(original_n)
            client_entropies.append(entropy)
            client_drifts.append(drift)
            client_losses.append(client_loss)
            client_accuracies.append(client_acc)
            client_epsilons.append(client_eps)
            client_histories[cid] = history
            epsilon_history[cid].append(float(client_eps))

            print(f"\nClient {cid} complete:")
            print(f"  Samples                  : {original_n}")
            print(f"  Final local loss         : {client_loss:.4f}")
            print(f"  Final local accuracy     : {client_acc:.4f}")
            print(f"  Cumulative epsilon       : {client_eps:.6f}")
            print(f"  Sigma                    : {client_sigmas[cid]:.6f}")
            print(f"  DP model drift           : {drift:.6e}")

            del client_loader
            aggressive_cuda_cleanup(f"client {cid} round {round_number}")

        # -------------------------------------------------------------
        # SERVER: SMART AGGREGATION + AUTOMATIC FEDYOGI
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print("SERVER: DP-PROTECTED SMART AGGREGATION + AUTOMATIC FEDYOGI")
        print("-" * 70)

        (
            new_global_state,
            smart_weights,
            server_diag,
        ) = server_optimizer.step(
            global_state=global_state,
            client_states=client_states,
            client_sample_counts=client_sample_counts,
            client_entropies=client_entropies,
            client_drifts=client_drifts,
        )

        for cid, count, weight, diag in zip(
            client_ids,
            client_sample_counts,
            smart_weights,
            server_optimizer.last_client_weight_diagnostics,
        ):
            print(
                f"  Client {cid}: N={count}, weight={weight:.6f}, "
                f"entropy={diag['entropy']:.4f}, "
                f"cosine={diag['cosine_reliability']:.4f}, "
                f"drift={diag['drift']:.3e}, "
                f"multiplier={diag['final_multiplier']:.4f}"
            )

        print(f"\nServer LR                 : {server_optimizer.server_lr:.6e}")
        print(f"Server LR factor          : {server_diag['lr_factor']:.6f}")
        print(f"Update magnitude ratio    : {server_diag['magnitude_ratio']:.6f}")
        print(f"Update volatility         : {server_diag['volatility']:.6f}")
        print(f"FedYogi beta1             : {server_optimizer.beta1:.6f}")
        print(f"FedYogi beta2             : {server_optimizer.beta2:.6f}")
        print(f"FedYogi update RMS        : {server_optimizer.last_update_rms:.6e}")
        print(f"FedYogi direction RMS     : {server_optimizer.last_direction_rms:.6e}")
        print(f"Applied server update RMS : {server_optimizer.last_applied_update_rms:.6e}")

        validate_state_dict(new_global_state, global_state)
        global_state = new_global_state
        global_model.load_state_dict(global_state, strict=True)

        # -------------------------------------------------------------
        # GLOBAL VALIDATION
        # -------------------------------------------------------------
        val_loss, val_acc, val_macro_f1, val_balanced_acc, _, _ = evaluate(
            global_model,
            val_loader,
            device,
            eval_criterion,
        )

        round_time = time.perf_counter() - round_start
        mean_client_loss = float(np.mean(client_losses))
        mean_client_acc = float(np.mean(client_accuracies))
        max_epsilon = float(max(client_epsilons))

        print("\n" + "=" * 78)
        print(f"ROUND {round_number} RESULTS")
        print("=" * 78)
        print(f"  Mean client loss        : {mean_client_loss:.4f}")
        print(f"  Mean client accuracy    : {mean_client_acc:.4f}")
        print(f"  Validation loss         : {val_loss:.4f}")
        print(f"  Validation accuracy     : {val_acc:.4f}")
        print(f"  Validation Macro-F1     : {val_macro_f1:.4f}")
        print(f"  Validation Balanced Acc : {val_balanced_acc:.4f}")
        print(f"  Maximum client epsilon  : {max_epsilon:.6f}")
        print(f"  Smart aggregation       : {dict(zip(client_ids, smart_weights))}")
        print(f"  Round time              : {round_time:.1f}s")

        log_rows.append(
            {
                "round": round_number,
                "mean_client_loss": mean_client_loss,
                "mean_client_acc": mean_client_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_balanced_accuracy": val_balanced_acc,
                "max_client_epsilon": max_epsilon,
                "epsilon_by_client": str(
                    {
                        int(cid): float(epsilon_history[cid][-1])
                        for cid in client_ids
                    }
                ),
                "sigma_by_client": str(
                    {
                        int(cid): float(client_sigmas[cid])
                        for cid in client_ids
                    }
                ),
                "smart_aggregation_weights": str(
                    {
                        int(cid): float(weight)
                        for cid, weight in zip(client_ids, smart_weights)
                    }
                ),
                "smart_aggregation_diagnostics": str(
                    server_optimizer.last_client_weight_diagnostics
                ),
                "client_entropies": str(
                    {
                        int(cid): float(entropy)
                        for cid, entropy in zip(client_ids, client_entropies)
                    }
                ),
                "client_drifts": str(
                    {
                        int(cid): float(drift)
                        for cid, drift in zip(client_ids, client_drifts)
                    }
                ),
                "fedyogi_server_lr": float(server_optimizer.server_lr),
                "fedyogi_beta1": float(server_optimizer.beta1),
                "fedyogi_beta2": float(server_optimizer.beta2),
                "fedyogi_lr_factor": float(server_diag["lr_factor"]),
                "fedyogi_update_magnitude_ratio": float(
                    server_diag["magnitude_ratio"]
                ),
                "fedyogi_update_volatility": float(
                    server_diag["volatility"]
                ),
                "fedyogi_update_rms": float(
                    server_optimizer.last_update_rms
                ),
                "fedyogi_direction_rms": float(
                    server_optimizer.last_direction_rms
                ),
                "fedyogi_applied_update_rms": float(
                    server_optimizer.last_applied_update_rms
                ),
                "client_epoch_histories": str(client_histories),
                "round_time_seconds": round_time,
            }
        )

        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        # -------------------------------------------------------------
        # FULL RESUME CHECKPOINT — EVERY COMPLETED ROUND
        # -------------------------------------------------------------
        config_snapshot = {
            "experiment": experiment_tag,
            "architecture": "MaxViT-Tiny V2 hybrid",
            "initialization": "fresh ImageNet pretrained",
            "partition": "HAM10000 lesion-level Dirichlet alpha=100.0",
            "partition_dir": ALPHA100_PARTITION_DIR,
            "client_sizes": list(EXPECTED_ALPHA100_CLIENT_SIZES),
            "normalization": "GroupNorm task heads; MaxViT LayerNorm",
            "dropout": False,
            "stochastic_depth": False,
            "augmentation": "existing on-the-fly training augmentation",
            "sampling": "Opacus Poisson",
            "grad_sample_mode": "ghost",
            "privacy_engine": "Opacus 1.6",
            "accountant": "RDP",
            "epsilon_target": target_epsilon,
            "delta": target_delta,
            "max_grad_norm": max_grad_norm,
            "sigma_by_client": client_sigmas,
            "sigma_metadata": sigma_metadata,
            "rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "clients_per_round": clients_per_round,
            "class_weights": FIXED_CLASS_WEIGHTS.detach().cpu().tolist(),
            "label_smoothing": label_smoothing,
            "fedprox_mu": fedprox_mu,
            "head_lr": head_lr,
            "backbone_lr": backbone_lr,
            "backbone_lr_ratio": backbone_lr_ratio,
            "weight_decay": weight_decay,
            "adaptive_client_lr": False,
            "adaptive_fedprox": False,
            "weighted_random_sampler": False,
            "two_x_sampling": False,
            "amp": False,
            "server_optimizer": "Automatic FedYogi",
            "client_aggregation": (
                "sqrt(N) + entropy + reliability + drift; "
                "bounded multiplier"
            ),
            "test_usage": "held-out test evaluated once after training",
            "checkpointing": "full resume checkpoint every completed round",
            "rng_checkpoint": True,
            "server_optimizer_checkpoint": True,
            "rdp_accountant_checkpoint": True,
            "aggressive_cuda_cleanup": True,
        }

        save_resume_checkpoint(
            resume_checkpoint_path,
            global_model,
            global_state,
            round_number,
            best_val_f1,
            best_val_acc,
            best_val_loss,
            best_round,
            server_optimizer,
            privacy_engines,
            client_sigmas,
            sigma_metadata,
            epsilon_history,
            log_rows,
            config_snapshot,
        )
        print(
            f"✓ FULL RESUME CHECKPOINT SAVED — completed round {round_number}"
        )

        # -------------------------------------------------------------
        # BEST CHECKPOINT
        # -------------------------------------------------------------
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_round = round_number

            config_snapshot = {
                "experiment": experiment_tag,
                "architecture": "MaxViT-Tiny V2 hybrid",
                "initialization": "fresh ImageNet pretrained",
                "partition": "HAM10000 lesion-level Dirichlet alpha=100.0",
            "partition_dir": ALPHA100_PARTITION_DIR,
            "client_sizes": list(EXPECTED_ALPHA100_CLIENT_SIZES),
                "normalization": "GroupNorm task heads; MaxViT LayerNorm",
                "dropout": False,
                "stochastic_depth": False,
                "augmentation": "existing on-the-fly training augmentation",
                "sampling": "Opacus Poisson",
                "grad_sample_mode": "ghost",
                "privacy_engine": "Opacus 1.6",
                "accountant": "RDP",
                "epsilon_target": target_epsilon,
                "delta": target_delta,
                "max_grad_norm": max_grad_norm,
                "sigma_by_client": client_sigmas,
                "sigma_metadata": sigma_metadata,
                "rounds": num_rounds,
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "clients_per_round": clients_per_round,
                "class_weights": FIXED_CLASS_WEIGHTS.detach().cpu().tolist(),
                "label_smoothing": label_smoothing,
                "fedprox_mu": fedprox_mu,
                "head_lr": head_lr,
                "backbone_lr": backbone_lr,
                "backbone_lr_ratio": backbone_lr_ratio,
                "weight_decay": weight_decay,
                "adaptive_client_lr": False,
                "adaptive_fedprox": False,
                "weighted_random_sampler": False,
                "two_x_sampling": False,
                "server_optimizer": "Automatic FedYogi",
                "client_aggregation": (
                    "sqrt(N) + entropy + reliability + drift; "
                    "bounded multiplier"
                ),
                "balanced_accuracy": "reported for validation and final test",
                "test_usage": "held-out test evaluated once after training",
            }

            privacy_snapshot = {
                "target_epsilon": target_epsilon,
                "target_delta": target_delta,
                "sigma_by_client": {
                    int(cid): float(client_sigmas[cid])
                    for cid in client_ids
                },
                "epsilon_by_client": {
                    int(cid): float(epsilon_history[cid][-1])
                    for cid in client_ids
                },
                "accountant": "RDP",
                "sampling": "Poisson",
                "max_grad_norm": max_grad_norm,
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
                privacy_snapshot,
            )

            print(
                f"\n✓ NEW BEST GLOBAL MODEL — "
                f"round {best_round}, "
                f"Val Macro-F1={best_val_f1:.4f}, "
                f"Val Acc={best_val_acc:.4f}"
            )

    # -----------------------------------------------------------------
    # FINAL MODEL / SERVER STATE
    # -----------------------------------------------------------------
    global_model.load_state_dict(global_state, strict=True)

    torch.save(
        {
            "model_state_dict": {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
            },
            "round": num_rounds,
            "experiment": experiment_tag,
            "best_round": best_round,
            "best_val_macro_f1": best_val_f1,
            "best_val_acc": best_val_acc,
            "server_optimizer_state": server_optimizer.state_dict(),
            "privacy_accountant_states": capture_all_privacy_states(privacy_engines),
            "rng_state": capture_rng_state(),
            "client_sigmas": client_sigmas,
            "sigma_metadata": sigma_metadata,
            "epsilon_history": epsilon_history,
            "target_epsilon": target_epsilon,
            "target_delta": target_delta,
        },
        final_model_path,
    )

    # -----------------------------------------------------------------
    # FINAL TEST: ONLY THE BEST VALIDATION CHECKPOINT
    # -----------------------------------------------------------------
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best checkpoint was not created:\n{best_model_path}"
        )

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )
    global_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    test_loss, test_acc, test_macro_f1, test_balanced_acc, test_preds, test_labels = evaluate(
        global_model,
        test_loader,
        device,
        eval_criterion,
    )

    report = classification_report(
        test_labels,
        test_preds,
        labels=list(range(config.NUM_CLASSES)),
        target_names=config.CLASS_LIST,
        zero_division=0,
    )

    class_scores = per_class_f1(test_labels, test_preds)
    pd.DataFrame(
        {
            "class": list(class_scores.keys()),
            "f1_score": list(class_scores.values()),
        }
    ).to_csv(per_class_path, index=False)

    save_confusion_matrix(
        test_labels,
        test_preds,
        confusion_path,
    )

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    plot_training_curves(
        pd.DataFrame(log_rows),
        curves_path,
    )

    final_epsilons = {
        int(cid): float(privacy_engines[cid].get_epsilon(target_delta))
        for cid in client_ids
    }

    print("\n" + "=" * 78)
    print("FINAL DP FEDERATED TEST RESULTS")
    print("=" * 78)
    print(f"Best federated round      : {best_round}")
    print(f"Best validation Macro-F1  : {best_val_f1:.6f}")
    print(f"Best validation accuracy  : {best_val_acc:.6f}")
    print(f"Test loss                 : {test_loss:.6f}")
    print(f"Test accuracy             : {test_acc:.6f}")
    print(f"Test Macro-F1             : {test_macro_f1:.6f}")
    print(f"Test Balanced Accuracy    : {test_balanced_acc:.6f}")
    print("\nFinal per-client epsilon:")
    for cid in client_ids:
        print(
            f"  Client {cid}: "
            f"epsilon={final_epsilons[cid]:.6f}, "
            f"sigma={client_sigmas[cid]:.6f}"
        )

    print("\nClassification Report:")
    print(report)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("FEDERATED MAXVIT-TINY V2 - DP-SGD TEST REPORT\n\n")
        f.write(f"Best federated round: {best_round}\n")
        f.write(f"Best validation Macro-F1: {best_val_f1:.6f}\n")
        f.write(f"Best validation Accuracy: {best_val_acc:.6f}\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Accuracy: {test_acc:.6f}\n")
        f.write(f"Test Macro-F1: {test_macro_f1:.6f}\n")
        f.write(f"Test Balanced Accuracy: {test_balanced_acc:.6f}\n")
        f.write(f"Target epsilon: {target_epsilon:.6f}\n")
        f.write(f"Target delta: {target_delta:.2e}\n")
        f.write(f"Best round: {best_round}\n\n")
        f.write("Sigma by client:\n")
        for cid in client_ids:
            f.write(
                f"Client {cid}: {client_sigmas[cid]:.8f}\n"
            )
        f.write("\nFinal epsilon by client:\n")
        for cid in client_ids:
            f.write(
                f"Client {cid}: {final_epsilons[cid]:.8f}\n"
            )
        f.write("\nClassification Report:\n\n")
        f.write(report)

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 78)
    print("FINAL DP EXPERIMENT COMPLETE")
    print("=" * 78)
    print(f"Total time                 : {elapsed / 60.0:.2f} minutes")
    print(f"Best model                 : {best_model_path}")
    print(f"Final model                : {final_model_path}")
    print(f"Latest resume checkpoint   : {resume_checkpoint_path}")
    print(f"Training log               : {log_path}")
    print(f"Test report                : {report_path}")
    print(f"Per-class F1               : {per_class_path}")
    print(f"Training curves            : {curves_path}")
    print(f"Confusion matrix           : {confusion_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
