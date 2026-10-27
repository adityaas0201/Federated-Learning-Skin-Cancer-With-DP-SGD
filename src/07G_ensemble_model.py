"""07G_ensemble_model.py

Late-fusion ensemble that preserves the two existing hybrid models unchanged.

Required sibling source files (normal project names):
    07A_model_convnext.py
    07F_model_maxvit_tiny.py

Expected inputs:
    images: [B, 3, config.IMAGE_SIZE, config.IMAGE_SIZE]
    handcrafted_features: [B, config.HANDCRAFTED_FEATURE_DIM] (1320)

Each original model independently performs its original feature extraction,
handcrafted branch (1320 -> 512 -> 256), feature concatenation, and classifier:
    ConvNeXt hybrid: [768 + 256] -> 512 -> 256 -> num_classes
    MaxViT hybrid:   [512 + 256] -> 512 -> 256 -> num_classes

Fusion: weighted mean of the two models' class probabilities, returned as log
probabilities. Probability fusion avoids one backbone dominating merely because
its raw logits have a larger numerical scale. The original branch architectures
and heads remain unchanged. Local training applies auxiliary cross-entropy to
each branch so both backbones learn directly.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent

# Match the config import convention used by the original model files.
CONFIG_DIR = HERE.parent / "configs"
if CONFIG_DIR.exists():
    sys.path.insert(0, str(CONFIG_DIR))
else:
    # Useful when this file is placed directly beside configs/ in another layout.
    CONFIG_DIR = HERE / "configs"
    if CONFIG_DIR.exists():
        sys.path.insert(0, str(CONFIG_DIR))

import config  # noqa: E402  (same config used by the original project)


def _load_source_module(module_name: str, candidate_names: tuple[str, ...]) -> ModuleType:
    """Load an original model source file by path, including numeric filenames."""
    candidates = [HERE / name for name in candidate_names]
    source_path = next((path for path in candidates if path.is_file()), None)
    if source_path is None:
        searched = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            f"Could not find source file for {module_name}. Searched:\n{searched}\n"
            "Keep 07G_ensemble_model.py beside the original ConvNeXt and MaxViT model files."
        )

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so normal Python module semantics are preserved.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_convnext_module = _load_source_module(
    "_project_convnext_model",
    (
        "07A_model_convnext.py",
        "07A_model_convnext(1).py",
        "Pasted text(20261016-113006).txt",
    ),
)
_maxvit_module = _load_source_module(
    "_project_maxvit_model",
    (
        "07F_model_maxvit_tiny.py",
        "07F_model_maxvit_tiny(2).py",
        "Pasted text (2).txt",
    ),
)


class ConvNeXtMaxViTEnsemble(nn.Module):
    """Jointly trainable late-fusion ensemble of the exact two hybrid models.

    Args:
        num_classes: Must match config.NUM_CLASSES (7 in this project).
        handcrafted_feature_dim: Must match config.HANDCRAFTED_FEATURE_DIM (1320).
        pretrained: Passes through to both original models, preserving their
            ImageNet-pretrained behavior when True.
        convnext_weight/maxvit_weight: Nonnegative logit-fusion weights. Defaults
            to equal weighting. They are normalized at inference/forward time.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        handcrafted_feature_dim: Optional[int] = None,
        pretrained: bool = True,
        convnext_weight: float = 1.0,
        maxvit_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes if num_classes is not None else config.NUM_CLASSES)
        self.handcrafted_feature_dim = int(
            handcrafted_feature_dim
            if handcrafted_feature_dim is not None
            else config.HANDCRAFTED_FEATURE_DIM
        )

        if self.num_classes != int(config.NUM_CLASSES):
            raise ValueError(
                f"num_classes={self.num_classes} does not match config.NUM_CLASSES="
                f"{config.NUM_CLASSES}"
            )
        if self.handcrafted_feature_dim != int(config.HANDCRAFTED_FEATURE_DIM):
            raise ValueError(
                "handcrafted_feature_dim does not match config: expected "
                f"{config.HANDCRAFTED_FEATURE_DIM}, got {self.handcrafted_feature_dim}"
            )
        if convnext_weight < 0 or maxvit_weight < 0:
            raise ValueError("Fusion weights must be nonnegative.")
        if convnext_weight + maxvit_weight == 0:
            raise ValueError("At least one fusion weight must be positive.")

        # These are the ORIGINAL complete model classes, not rewritten versions.
        self.convnext_model = _convnext_module.HybridClassifier(
            num_classes=self.num_classes,
            handcrafted_feature_dim=self.handcrafted_feature_dim,
            pretrained=pretrained,
        )
        self.maxvit_model = _maxvit_module.HybridClassifier(
            num_classes=self.num_classes,
            handcrafted_feature_dim=self.handcrafted_feature_dim,
            pretrained=pretrained,
        )

        weight_sum = float(convnext_weight + maxvit_weight)
        self.convnext_weight = float(convnext_weight / weight_sum)
        self.maxvit_weight = float(maxvit_weight / weight_sum)

    def forward(
        self,
        images: torch.Tensor,
        handcrafted_features: torch.Tensor,
        return_aux: bool = False,
    ):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], received {tuple(images.shape)}")
        if handcrafted_features.ndim != 2:
            raise ValueError(
                "Expected handcrafted_features [B,F], received "
                f"{tuple(handcrafted_features.shape)}"
            )
        if images.shape[0] != handcrafted_features.shape[0]:
            raise ValueError("Image and handcrafted-feature batch sizes do not match.")
        if handcrafted_features.shape[1] != self.handcrafted_feature_dim:
            raise ValueError(
                f"Expected {self.handcrafted_feature_dim} handcrafted features, got "
                f"{handcrafted_features.shape[1]}"
            )

        # Both full models run their unchanged original branches and classifiers.
        convnext_logits = self.convnext_model(images, handcrafted_features)
        maxvit_logits = self.maxvit_model(images, handcrafted_features)

        # Probability-level late fusion. Raw logits from different architectures
        # can have very different scales; averaging them directly may let one
        # branch dominate the ensemble. Return log probabilities so existing
        # CrossEntropyLoss/evaluation code remains compatible.
        convnext_probs = torch.softmax(convnext_logits.float(), dim=1)
        maxvit_probs = torch.softmax(maxvit_logits.float(), dim=1)
        fused_probs = (
            self.convnext_weight * convnext_probs
            + self.maxvit_weight * maxvit_probs
        ).clamp_min(1e-8)
        fused_probs = fused_probs / fused_probs.sum(dim=1, keepdim=True)
        logits = fused_probs.log()

        if return_aux:
            return logits, {
                "convnext_logits": convnext_logits,
                "maxvit_logits": maxvit_logits,
                "convnext_probs": convnext_probs,
                "maxvit_probs": maxvit_probs,
                "fused_probs": fused_probs,
                "fusion_weights": {
                    "convnext": self.convnext_weight,
                    "maxvit": self.maxvit_weight,
                },
            }
        return logits

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_model(pretrained: bool = True) -> ConvNeXtMaxViTEnsemble:
    """Build using the project's existing config values."""
    return ConvNeXtMaxViTEnsemble(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=config.HANDCRAFTED_FEATURE_DIM,
        pretrained=pretrained,
    )


# Aliases for convenient integration with existing project scripts.
EnsembleClassifier = ConvNeXtMaxViTEnsemble
HybridClassifier = ConvNeXtMaxViTEnsemble


def smoke_test(batch_size: Optional[int] = None, pretrained: bool = False) -> None:
    """Run forward/backward checks; use batch_size=4 on the target RTX 3050.

    The current authoring environment may not have CUDA. On a CUDA machine this
    test reports peak allocated/reserved memory and re-raises any CUDA OOM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = 4 if device.type == "cuda" else 1

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(pretrained=pretrained).to(device).train()
    images = torch.randn(
        batch_size, 3, int(config.IMAGE_SIZE), int(config.IMAGE_SIZE), device=device
    )
    handcrafted = torch.randn(
        batch_size, int(config.HANDCRAFTED_FEATURE_DIM), device=device
    )
    labels = torch.randint(0, int(config.NUM_CLASSES), (batch_size,), device=device)

    # AMP reduces activation memory on supported CUDA hardware; CPU uses FP32.
    amp_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        logits, aux = model(images, handcrafted, return_aux=True)
        loss = nn.CrossEntropyLoss()(logits.float(), labels)

    if tuple(logits.shape) != (batch_size, int(config.NUM_CLASSES)):
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if tuple(aux["convnext_logits"].shape) != (batch_size, int(config.NUM_CLASSES)):
        raise AssertionError("ConvNeXt output shape check failed.")
    if tuple(aux["maxvit_logits"].shape) != (batch_size, int(config.NUM_CLASSES)):
        raise AssertionError("MaxViT output shape check failed.")

    loss.backward()

    checks = {
        "ConvNeXt backbone gradient": any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.convnext_model.backbone.parameters()
            if p.requires_grad
        ),
        "ConvNeXt handcrafted gradient": any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.convnext_model.handcrafted_branch.parameters()
        ),
        "ConvNeXt classifier gradient": any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.convnext_model.classifier.parameters()
        ),
        "MaxViT backbone gradient": any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.maxvit_model.backbone.parameters()
            if p.requires_grad
        ),
        "MaxViT handcrafted gradient": any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.maxvit_model.handcrafted_branch.parameters()
        ),
        "MaxViT classifier gradient": any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.maxvit_model.classifier.parameters()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Gradient-flow verification failed: {checks}")

    print(f"Device: {device}; batch size: {batch_size}; pretrained: {pretrained}")
    print(f"Logits: {tuple(logits.shape)}; loss: {loss.item():.5f}")
    print(f"Parameters: {model.parameter_counts()}")
    print(f"Gradient checks: {checks}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Peak allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB")
        print(f"Peak reserved:  {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GiB")


if __name__ == "__main__":
    smoke_test()
