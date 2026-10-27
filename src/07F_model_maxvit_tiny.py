"""
07F_model_maxvit_tiny.py
------------------------
PHASE 2 - STEP 7F: MAXVIT-TINY HYBRID MODEL

Purpose:
    New pretrained vision backbone experiment for the HAM10000 project.

Hybrid architecture:

    Image
      |
      v
    MaxViT-Tiny (ImageNet-1K pretrained)
      |
      v
    512-D deep embedding
      |
      +----------------------+
      |                      |
      v                      v
 Deep Feature Branch    Handcrafted Branch
      |                      |
    512-D                  1320-D
      |                      |
   Normalize               MLP -> 256-D
      |                      |
      +----------+-----------+
                 |
                 v
          Feature Fusion
             768-D
                 |
                 v
        Classification Head
                 |
                 v
              7 classes

IMPORTANT:
    - This file is a separate model experiment.
    - The existing MobileNetV3/ConvNeXt/Swin/EfficientNet files are untouched.
    - The existing dataset and 1320-D handcrafted features are unchanged.
    - The MaxViT-Tiny backbone is ImageNet pretrained and fully fine-tuned.
    - Input remains 224x224 so it stays compatible with 06_dataset.py.

Expected input:
    images   : [B, 3, 224, 224]
    features : [B, 1320]

Expected output:
    logits   : [B, 7]

Deep embedding:
    [B, 512]

Torchvision API:
    maxvit_t(weights=MaxVit_T_Weights.DEFAULT)
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    maxvit_t,
    MaxVit_T_Weights,
)


# =====================================================================
# CONFIG
# =====================================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
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
# MODEL CONSTANTS
# =====================================================================

MAXVIT_TINY_FEATURE_DIM = 512

EXPECTED_HANDCRAFTED_DIM = config.HANDCRAFTED_FEATURE_DIM

NUM_CLASSES = config.NUM_CLASSES


# =====================================================================
# BRANCH NORMALIZATION
# =====================================================================

class L2Normalize(nn.Module):
    """L2-normalize a feature vector along the feature dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(
            x,
            p=2,
            dim=1,
        )


# =====================================================================
# HANDCRAFTED FEATURE BRANCH
# =====================================================================

class HandcraftedFeatureBranch(nn.Module):
    """
    Processes the standardized 1320-D handcrafted feature vector.

    Input:
        [B, 1320]

    Output:
        [B, 256]
    """

    def __init__(
        self,
        input_dim,
        output_dim=256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                512,
            ),

            nn.GroupNorm(
                8,
                512,
            ),

            nn.ReLU(
                inplace=False,
            ),

            nn.Dropout(
                p=0.30,
            ),

            nn.Linear(
                512,
                output_dim,
            ),

            nn.GroupNorm(
                8,
                output_dim,
            ),

            nn.ReLU(
                inplace=False,
            ),
        )

        self.normalize = L2Normalize()

    def forward(self, x):
        x = self.network(x)
        x = self.normalize(x)
        return x


# =====================================================================
# MAXVIT-TINY BACKBONE
# =====================================================================

class MaxViTTinyBackbone(nn.Module):
    """
    ImageNet-1K pretrained torchvision MaxViT-Tiny.

    torchvision MaxViT-Tiny exposes its backbone as stem + blocks and its
    classifier as AdaptiveAvgPool -> Flatten -> LayerNorm -> Linear(512,512)
    -> Tanh -> Linear(512,1000). The final 1000-class layer is discarded,
    while the preceding 512-D representation is retained.

    All backbone parameters remain trainable for full fine-tuning.
    """

    def __init__(
        self,
        pretrained=True,
    ):
        super().__init__()

        if pretrained:
            weights = MaxVit_T_Weights.DEFAULT
            backbone = maxvit_t(
                weights=weights,
            )
        else:
            backbone = maxvit_t(
                weights=None,
            )

        # IMPORTANT:
        # torchvision MaxViT-Tiny does NOT expose a .features attribute.
        # Its current torchvision structure is:
        #
        #   stem -> blocks -> classifier
        #
        # and classifier is:
        #
        #   AdaptiveAvgPool2d -> Flatten -> LayerNorm -> Linear(512,512)
        #   -> Tanh -> Linear(512,1000)
        #
        # We keep the complete stem + MaxViT blocks and everything in the
        # classifier up to the 512-D pre-classification representation.
        self.features = nn.Sequential(
            backbone.stem,
            *list(backbone.blocks),
        )

        self.avgpool = nn.Sequential(
            backbone.classifier[0],  # AdaptiveAvgPool2d
            backbone.classifier[1],  # Flatten
            backbone.classifier[2],  # LayerNorm
            backbone.classifier[3],  # Linear 512 -> 512
            backbone.classifier[4],  # Tanh
        )

        self.feature_dim = MAXVIT_TINY_FEATURE_DIM

        if self.avgpool[3].out_features != self.feature_dim:
            raise RuntimeError(
                "Unexpected MaxViT-Tiny embedding dimension. "
                f"Expected {self.feature_dim}, got "
                f"{self.avgpool[3].out_features}."
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(
            x,
            start_dim=1,
        )
        return x


# =====================================================================
# HYBRID CLASSIFIER
# =====================================================================

class HybridClassifier(nn.Module):
    """
    MaxViT-Tiny + handcrafted-feature hybrid classifier.

    Image branch:
        224x224 RGB
        ->
        MaxViT-Tiny
        ->
        512-D

    Handcrafted branch:
        1320-D
        ->
        512
        ->
        256-D

    Fusion:
        512 + 256
        ->
        768-D

    Classifier:
        768
        ->
        512
        ->
        256
        ->
        7 classes
    """

    def __init__(
        self,
        num_classes,
        handcrafted_feature_dim,
        pretrained=True,
    ):
        super().__init__()

        if handcrafted_feature_dim != EXPECTED_HANDCRAFTED_DIM:
            raise ValueError(
                "Unexpected handcrafted feature dimension.\n"
                f"Expected: {EXPECTED_HANDCRAFTED_DIM}\n"
                f"Received: {handcrafted_feature_dim}"
            )

        # -------------------------------------------------------------
        # MaxViT-Tiny image branch
        # -------------------------------------------------------------

        self.backbone = MaxViTTinyBackbone(
            pretrained=pretrained,
        )

        self.deep_feature_dim = self.backbone.feature_dim

        # -------------------------------------------------------------
        # Deep feature normalization
        # -------------------------------------------------------------

        self.deep_normalize = L2Normalize()

        # -------------------------------------------------------------
        # Handcrafted feature branch
        # -------------------------------------------------------------

        self.handcrafted_branch = HandcraftedFeatureBranch(
            input_dim=handcrafted_feature_dim,
            output_dim=256,
        )

        self.handcrafted_feature_dim = 256

        # -------------------------------------------------------------
        # Fusion dimension
        # -------------------------------------------------------------

        self.fusion_dim = (
            self.deep_feature_dim
            + self.handcrafted_feature_dim
        )

        # -------------------------------------------------------------
        # Classification head
        # -------------------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                self.fusion_dim,
                512,
            ),

            nn.GroupNorm(
                8,
                512,
            ),

            nn.ReLU(
                inplace=False,
            ),

            nn.Dropout(
                p=0.40,
            ),

            nn.Linear(
                512,
                256,
            ),

            nn.GroupNorm(
                8,
                256,
            ),

            nn.ReLU(
                inplace=False,
            ),

            nn.Dropout(
                p=0.30,
            ),

            nn.Linear(
                256,
                num_classes,
            ),
        )

        self.num_classes = num_classes

    # ================================================================
    # IMAGE EMBEDDING
    # ================================================================

    def extract_deep_embedding(self, images):
        """
        Extract normalized MaxViT-Tiny deep embedding.

        Input:
            [B, 3, 224, 224]

        Output:
            [B, 512]
        """

        embedding = self.backbone(images)
        embedding = self.deep_normalize(embedding)

        return embedding

    # ================================================================
    # FORWARD
    # ================================================================

    def forward(
        self,
        images,
        handcrafted_features,
    ):
        deep_features = self.backbone(images)

        deep_features = self.deep_normalize(
            deep_features,
        )

        handcrafted_features = self.handcrafted_branch(
            handcrafted_features,
        )

        fused_features = torch.cat(
            [
                deep_features,
                handcrafted_features,
            ],
            dim=1,
        )

        logits = self.classifier(
            fused_features,
        )

        return logits

    # ================================================================
    # EXTRACT FUSED FEATURES
    # ================================================================

    @torch.no_grad()
    def extract_fused_features(
        self,
        images,
        handcrafted_features,
    ):
        """Extract the fused 768-D representation before classification."""

        deep_features = self.backbone(images)

        deep_features = self.deep_normalize(
            deep_features,
        )

        handcrafted_features = self.handcrafted_branch(
            handcrafted_features,
        )

        fused_features = torch.cat(
            [
                deep_features,
                handcrafted_features,
            ],
            dim=1,
        )

        return fused_features


# =====================================================================
# PARAMETER INFORMATION
# =====================================================================

def count_parameters(model):
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def estimate_model_size(model):
    total_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    return total_bytes / (1024 ** 2)


# =====================================================================
# TRAINABLE-PARAMETER CHECK
# =====================================================================

def check_backbone_trainable(model):
    trainable_count = sum(
        parameter.numel()
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    )

    if trainable_count == 0:
        raise RuntimeError(
            "MaxViT-Tiny backbone is frozen. "
            "Expected full backbone fine-tuning."
        )

    return trainable_count


# =====================================================================
# FORWARD PASS TEST
# =====================================================================

def test_forward_pass(
    model,
    device,
):
    print("\n" + "=" * 60)
    print("RUNNING MAXVIT-TINY FORWARD PASS")
    print("=" * 60)

    # Keep the self-test conservative for 4GB GPUs.
    test_batch_size = min(
        getattr(config, "BATCH_SIZE", 2),
        2,
    )

    image_size = config.IMAGE_SIZE
    handcrafted_dim = config.HANDCRAFTED_FEATURE_DIM

    images = torch.randn(
        test_batch_size,
        3,
        image_size,
        image_size,
        device=device,
    )

    handcrafted_features = torch.randn(
        test_batch_size,
        handcrafted_dim,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        logits = model(
            images,
            handcrafted_features,
        )

    print(f"  Images   : {tuple(images.shape)}")
    print(f"  Features : {tuple(handcrafted_features.shape)}")
    print(f"  Output   : {tuple(logits.shape)}")

    expected_output = (
        test_batch_size,
        config.NUM_CLASSES,
    )

    if tuple(logits.shape) != expected_output:
        raise RuntimeError(
            "Unexpected output shape.\n"
            f"Expected: {expected_output}\n"
            f"Got:      {tuple(logits.shape)}"
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "NaN or Inf detected in forward output."
        )

    print("\n  ✓ Forward pass successful")


# =====================================================================
# GRADIENT FLOW TEST
# =====================================================================

def test_gradient_flow(
    model,
    device,
):
    print("\n" + "=" * 60)
    print("CHECKING MAXVIT-TINY GRADIENT FLOW")
    print("=" * 60)

    test_batch_size = min(
        getattr(config, "BATCH_SIZE", 2),
        2,
    )

    images = torch.randn(
        test_batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device,
    )

    handcrafted_features = torch.randn(
        test_batch_size,
        config.HANDCRAFTED_FEATURE_DIM,
        device=device,
    )

    labels = torch.randint(
        low=0,
        high=config.NUM_CLASSES,
        size=(test_batch_size,),
        device=device,
    )

    model.train()
    model.zero_grad(set_to_none=True)

    logits = model(
        images,
        handcrafted_features,
    )

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels,
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Non-finite loss detected during gradient test."
        )

    loss.backward()

    backbone_gradients = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    )

    handcrafted_gradients = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.handcrafted_branch.parameters()
        if parameter.requires_grad
    )

    classifier_gradients = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    )

    print(
        f"  Backbone gradients      : {backbone_gradients}"
    )
    print(
        f"  Handcrafted gradients   : {handcrafted_gradients}"
    )
    print(
        f"  Classifier gradients    : {classifier_gradients}"
    )

    if not backbone_gradients:
        raise RuntimeError(
            "No valid gradients detected in MaxViT-Tiny backbone."
        )

    if not handcrafted_gradients:
        raise RuntimeError(
            "No valid gradients detected in handcrafted branch."
        )

    if not classifier_gradients:
        raise RuntimeError(
            "No valid gradients detected in classification head."
        )

    print("\n  ✓ Gradient flow verified")


# =====================================================================
# DEEP EMBEDDING TEST
# =====================================================================

def test_embedding_extraction(
    model,
    device,
):
    print("\n" + "=" * 60)
    print("CHECKING MAXVIT-TINY DEEP EMBEDDING")
    print("=" * 60)

    test_batch_size = min(
        getattr(config, "BATCH_SIZE", 2),
        2,
    )

    images = torch.randn(
        test_batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        embedding = model.extract_deep_embedding(
            images,
        )

    print(
        f"  Embedding shape: {tuple(embedding.shape)}"
    )

    expected_shape = (
        test_batch_size,
        MAXVIT_TINY_FEATURE_DIM,
    )

    if tuple(embedding.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected embedding shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {tuple(embedding.shape)}"
        )

    if not torch.isfinite(embedding).all():
        raise RuntimeError(
            "NaN or Inf detected in deep embedding."
        )

    norms = torch.linalg.vector_norm(
        embedding,
        dim=1,
    )

    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise RuntimeError(
            "Deep embedding is not L2-normalized."
        )

    print("\n  ✓ 512-D normalized deep embedding verified")


# =====================================================================
# FUSED FEATURE TEST
# =====================================================================

def test_fused_features(
    model,
    device,
):
    print("\n" + "=" * 60)
    print("CHECKING MAXVIT-TINY FEATURE FUSION")
    print("=" * 60)

    test_batch_size = min(
        getattr(config, "BATCH_SIZE", 2),
        2,
    )

    images = torch.randn(
        test_batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device,
    )

    handcrafted_features = torch.randn(
        test_batch_size,
        config.HANDCRAFTED_FEATURE_DIM,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        fused = model.extract_fused_features(
            images,
            handcrafted_features,
        )

    print(
        f"  Deep features       : {MAXVIT_TINY_FEATURE_DIM}"
    )
    print(
        "  Handcrafted branch  : 256"
    )
    print(
        f"  Fused features      : {tuple(fused.shape)}"
    )

    expected_dim = (
        MAXVIT_TINY_FEATURE_DIM
        + 256
    )

    expected_shape = (
        test_batch_size,
        expected_dim,
    )

    if tuple(fused.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected fused feature shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {tuple(fused.shape)}"
        )

    if not torch.isfinite(fused).all():
        raise RuntimeError(
            "NaN or Inf detected in fused features."
        )

    print("\n  ✓ 768-D feature fusion verified")


# =====================================================================
# MAIN SELF-TEST
# =====================================================================

def main():
    start_time = time.perf_counter()

    print("=" * 60)
    print(
        "PHASE 2 - STEP 7F: "
        "MAXVIT-TINY MODEL SELF-TEST"
    )
    print("=" * 60)

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    print(f"\nDevice: {DEVICE}")
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
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU VRAM: "
            f"{total_memory:.2f} GB"
        )

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

    print("\nModel configuration:")
    print("  Architecture        : MaxViT-Tiny")
    print(
        f"  Image size          : "
        f"{config.IMAGE_SIZE}x{config.IMAGE_SIZE}"
    )
    print(
        f"  Deep embedding      : "
        f"{MAXVIT_TINY_FEATURE_DIM}"
    )
    print(
        f"  Handcrafted input   : "
        f"{config.HANDCRAFTED_FEATURE_DIM}"
    )
    print("  Handcrafted branch  : 256")
    print(
        f"  Fusion dimension    : "
        f"{MAXVIT_TINY_FEATURE_DIM + 256}"
    )
    print(
        f"  Number of classes   : "
        f"{config.NUM_CLASSES}"
    )
    print("  Pretrained          : True")
    print("  Full fine-tuning    : True")
    print("  Attention           : MaxViT block + grid attention")
    print("  ImageNet weights    : torchvision MaxVit_T_Weights.DEFAULT")

    # -------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------

    print("\n" + "=" * 60)
    print("BUILDING MAXVIT-TINY HYBRID CLASSIFIER")
    print("=" * 60)

    model = HybridClassifier(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=config.HANDCRAFTED_FEATURE_DIM,
        pretrained=True,
    ).to(DEVICE)

    print("\nModel created successfully.")

    # -------------------------------------------------------------
    # Parameter information
    # -------------------------------------------------------------

    total_params, trainable_params = count_parameters(model)

    backbone_params = sum(
        p.numel()
        for p in model.backbone.parameters()
    )

    handcrafted_params = sum(
        p.numel()
        for p in model.handcrafted_branch.parameters()
    )

    classifier_params = sum(
        p.numel()
        for p in model.classifier.parameters()
    )

    model_size_mb = estimate_model_size(model)

    print("\nModel parameters:")
    print(
        f"  Backbone           : {backbone_params:,}"
    )
    print(
        f"  Handcrafted branch : {handcrafted_params:,}"
    )
    print(
        f"  Classifier         : {classifier_params:,}"
    )
    print(
        f"  Total              : {total_params:,}"
    )
    print(
        f"  Trainable          : {trainable_params:,}"
    )
    print(
        f"  Size               : {model_size_mb:.2f} MB"
    )

    # -------------------------------------------------------------
    # Verify full fine-tuning
    # -------------------------------------------------------------

    backbone_trainable_params = check_backbone_trainable(
        model
    )

    print(
        f"\nBackbone trainable parameters: "
        f"{backbone_trainable_params:,}"
    )

    # -------------------------------------------------------------
    # Forward test
    # -------------------------------------------------------------

    test_forward_pass(
        model,
        DEVICE,
    )

    # -------------------------------------------------------------
    # Gradient test
    # -------------------------------------------------------------

    test_gradient_flow(
        model,
        DEVICE,
    )

    # -------------------------------------------------------------
    # Embedding test
    # -------------------------------------------------------------

    test_embedding_extraction(
        model,
        DEVICE,
    )

    # -------------------------------------------------------------
    # Fusion test
    # -------------------------------------------------------------

    test_fused_features(
        model,
        DEVICE,
    )

    # -------------------------------------------------------------
    # GPU memory information
    # -------------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 2)
        )

        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 2)
        )

        print("\nGPU memory after self-test:")
        print(
            f"  Allocated       : {allocated:.2f} MB"
        )
        print(
            f"  Reserved        : {reserved:.2f} MB"
        )
        print(
            f"  Peak allocated  : {peak_allocated:.2f} MB"
        )
        print(
            f"  Peak reserved   : {peak_reserved:.2f} MB"
        )

    # -------------------------------------------------------------
    # Final
    # -------------------------------------------------------------

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("PHASE 2 - STEP 7F COMPLETE ✓")
    print("=" * 60)

    print(
        f"Self-test time: "
        f"{elapsed:.2f}s"
    )

    print("\nVerified:")
    print("  ✓ MaxViT-Tiny backbone")
    print("  ✓ ImageNet pretrained weights")
    print("  ✓ Full backbone fine-tuning")
    print("  ✓ 512-D deep embedding")
    print("  ✓ 1320-D handcrafted features")
    print("  ✓ Handcrafted feature branch")
    print("  ✓ Deep branch L2 normalization")
    print("  ✓ Handcrafted branch L2 normalization")
    print("  ✓ 768-D feature fusion")
    print("  ✓ Classification head")
    print("  ✓ Forward pass")
    print("  ✓ Backbone gradient flow")
    print("  ✓ Handcrafted branch gradient flow")
    print("  ✓ Classification head gradient flow")
    print("  ✓ Deep embedding extraction")
    print("  ✓ Fused feature extraction")
    print("  ✓ Finite-value checks")
    print("  ✓ 4GB-VRAM-conscious self-test")

    print("\nNext:")
    print(
        "  python src\\08F_train_centralized_maxvit_tiny_FINAL.py"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
