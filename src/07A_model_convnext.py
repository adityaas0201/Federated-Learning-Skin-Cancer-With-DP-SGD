"""
07A_model_convnext.py
---------------------
PHASE 2 - STEP 7A: CONVNEXT-TINY HYBRID MODEL

Purpose:
    Alternative deep-learning backbone for comparison with
    MobileNetV3-Small.

Hybrid architecture:

    Image
      |
      v
    ConvNeXt-Tiny
      |
      v
    768-D deep embedding
      |
      +----------------------+
      |                      |
      v                      v
 Deep Feature Branch    Handcrafted Branch
      |                      |
   768-D                  1320-D
      |                      |
 Normalize               Normalize
      |                      |
      +----------+-----------+
                 |
                 v
          Feature Fusion
                 |
                 v
        Classification Head
                 |
                 v
              7 classes

IMPORTANT:
    - This file does NOT replace 07_model.py.
    - 07_model.py remains the MobileNetV3-Small model.
    - This model is used as a separate comparison experiment.
    - The dataset and handcrafted features remain unchanged.

Expected input:
    images   : [B, 3, 224, 224]
    features : [B, 1320]

Expected output:
    logits   : [B, 7]

Deep embedding:
    [B, 768]

Compatible with:
    06_dataset.py
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


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
# MODEL CONSTANTS
# =====================================================================

CONVNEXT_TINY_FEATURE_DIM = 768

EXPECTED_HANDCRAFTED_DIM = config.HANDCRAFTED_FEATURE_DIM

NUM_CLASSES = config.NUM_CLASSES


# =====================================================================
# BRANCH NORMALIZATION
# =====================================================================

class L2Normalize(nn.Module):
    """
    L2-normalizes a feature vector along the feature dimension.

    This prevents one branch from dominating the other purely
    because of feature magnitude.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(
            x,
            p=2,
            dim=1
        )


# =====================================================================
# HANDCRAFTED FEATURE BRANCH
# =====================================================================

class HandcraftedFeatureBranch(nn.Module):
    """
    Processes the 1320-D standardized handcrafted features.

    Input:
        [B, 1320]

    Output:
        [B, 256]
    """

    def __init__(
        self,
        input_dim,
        output_dim=256
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                512
            ),

            nn.GroupNorm(
                8,
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                512,
                output_dim
            ),

            nn.GroupNorm(
                8,
                output_dim
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        self.normalize = L2Normalize()

    def forward(self, x):

        x = self.network(x)

        x = self.normalize(x)

        return x


# =====================================================================
# CONVNEXT-TINY BACKBONE
# =====================================================================

class ConvNeXtTinyBackbone(nn.Module):
    """
    ConvNeXt-Tiny pretrained on ImageNet.

    torchvision ConvNeXt-Tiny produces a 768-D final feature
    representation.

    The classifier is removed because we use ConvNeXt only as
    a feature extractor.

    IMPORTANT:
        requires_grad remains True for all backbone parameters,
        therefore the entire backbone is fine-tuned.
    """

    def __init__(
        self,
        pretrained=True
    ):
        super().__init__()

        if pretrained:

            weights = ConvNeXt_Tiny_Weights.DEFAULT

            backbone = convnext_tiny(
                weights=weights
            )

        else:

            backbone = convnext_tiny(
                weights=None
            )

        # ------------------------------------------------------------
        # Keep ConvNeXt feature extractor.
        #
        # torchvision ConvNeXt structure:
        #
        # features -> avgpool -> classifier
        #
        # We remove the original classifier.
        # ------------------------------------------------------------

        self.features = backbone.features

        self.avgpool = backbone.avgpool

        self.feature_dim = CONVNEXT_TINY_FEATURE_DIM

    def forward(self, x):

        # ConvNeXt feature maps
        x = self.features(x)

        # Global average pooling
        x = self.avgpool(x)

        # Flatten
        x = torch.flatten(
            x,
            start_dim=1
        )

        return x


# =====================================================================
# HYBRID CLASSIFIER
# =====================================================================

class HybridClassifier(nn.Module):
    """
    Hybrid ConvNeXt-Tiny + handcrafted feature classifier.

    Image branch:
        224x224 RGB image
        ->
        ConvNeXt-Tiny
        ->
        768-D deep embedding

    Handcrafted branch:
        1320-D
        ->
        256-D representation

    Fusion:
        768 + 256
        ->
        1024-D fused representation

    Classifier:
        1024
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
        pretrained=True
    ):
        super().__init__()

        # ------------------------------------------------------------
        # Validate dimensions
        # ------------------------------------------------------------

        if handcrafted_feature_dim != EXPECTED_HANDCRAFTED_DIM:

            raise ValueError(
                f"Unexpected handcrafted feature dimension.\n"
                f"Expected: {EXPECTED_HANDCRAFTED_DIM}\n"
                f"Received: {handcrafted_feature_dim}"
            )

        # ------------------------------------------------------------
        # ConvNeXt-Tiny image branch
        # ------------------------------------------------------------

        self.backbone = ConvNeXtTinyBackbone(
            pretrained=pretrained
        )

        self.deep_feature_dim = (
            self.backbone.feature_dim
        )

        # ------------------------------------------------------------
        # Deep feature normalization
        # ------------------------------------------------------------

        self.deep_normalize = L2Normalize()

        # ------------------------------------------------------------
        # Handcrafted branch
        # ------------------------------------------------------------

        self.handcrafted_branch = (
            HandcraftedFeatureBranch(
                input_dim=handcrafted_feature_dim,
                output_dim=256
            )
        )

        self.handcrafted_feature_dim = 256

        # ------------------------------------------------------------
        # Fusion dimension
        # ------------------------------------------------------------

        self.fusion_dim = (
            self.deep_feature_dim
            +
            self.handcrafted_feature_dim
        )

        # ------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                self.fusion_dim,
                512
            ),

            nn.GroupNorm(
                8,
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.40
            ),

            nn.Linear(
                512,
                256
            ),

            nn.GroupNorm(
                8,
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                256,
                num_classes
            )
        )

        self.num_classes = num_classes

    # ================================================================
    # IMAGE EMBEDDING
    # ================================================================

    def extract_deep_embedding(
        self,
        images
    ):
        """
        Extract ConvNeXt-Tiny 768-D embedding.

        Input:
            [B, 3, 224, 224]

        Output:
            [B, 768]
        """

        embedding = self.backbone(
            images
        )

        embedding = self.deep_normalize(
            embedding
        )

        return embedding

    # ================================================================
    # FORWARD
    # ================================================================

    def forward(
        self,
        images,
        handcrafted_features
    ):
        """
        Forward pass.

        Args:
            images:
                [B, 3, 224, 224]

            handcrafted_features:
                [B, 1320]

        Returns:
            logits:
                [B, 7]
        """

        # ------------------------------------------------------------
        # Deep image representation
        # ------------------------------------------------------------

        deep_features = self.backbone(
            images
        )

        deep_features = self.deep_normalize(
            deep_features
        )

        # ------------------------------------------------------------
        # Handcrafted representation
        # ------------------------------------------------------------

        handcrafted_features = (
            self.handcrafted_branch(
                handcrafted_features
            )
        )

        # ------------------------------------------------------------
        # Feature fusion
        # ------------------------------------------------------------

        fused_features = torch.cat(
            [
                deep_features,
                handcrafted_features
            ],
            dim=1
        )

        # ------------------------------------------------------------
        # Classification
        # ------------------------------------------------------------

        logits = self.classifier(
            fused_features
        )

        return logits

    # ================================================================
    # EXTRACT FUSED FEATURES
    # ================================================================

    @torch.no_grad()
    def extract_fused_features(
        self,
        images,
        handcrafted_features
    ):
        """
        Extract the fused representation before the classifier.

        Output:
            [B, 1024]
        """

        deep_features = self.backbone(
            images
        )

        deep_features = self.deep_normalize(
            deep_features
        )

        handcrafted_features = (
            self.handcrafted_branch(
                handcrafted_features
            )
        )

        fused_features = torch.cat(
            [
                deep_features,
                handcrafted_features
            ],
            dim=1
        )

        return fused_features


# =====================================================================
# MODEL PARAMETER INFORMATION
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

    return total_bytes / (
        1024 ** 2
    )


# =====================================================================
# CHECK TRAINABLE PARAMETERS
# =====================================================================

def check_backbone_trainable(
    model
):

    backbone_trainable = any(
        parameter.requires_grad
        for parameter in model.backbone.parameters()
    )

    if not backbone_trainable:

        raise RuntimeError(
            "ConvNeXt-Tiny backbone is frozen.\n"
            "Expected full backbone fine-tuning."
        )

    return backbone_trainable


# =====================================================================
# FORWARD PASS TEST
# =====================================================================

def test_forward_pass(
    model,
    device
):

    print(
        "\n" + "=" * 60
    )

    print(
        "RUNNING FORWARD PASS"
    )

    print(
        "=" * 60
    )

    # ------------------------------------------------------------
    # Use a smaller batch on limited VRAM.
    # ------------------------------------------------------------

    test_batch_size = min(
        getattr(config, "BATCH_SIZE", 8),
        8
    )

    image_size = config.IMAGE_SIZE

    handcrafted_dim = (
        config.HANDCRAFTED_FEATURE_DIM
    )

    images = torch.randn(
        test_batch_size,
        3,
        image_size,
        image_size,
        device=device
    )

    handcrafted_features = torch.randn(
        test_batch_size,
        handcrafted_dim,
        device=device
    )

    print(
        f"  Images   : {tuple(images.shape)}"
    )

    print(
        f"  Features : "
        f"{tuple(handcrafted_features.shape)}"
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            images,
            handcrafted_features
        )

    print(
        f"  Output   : "
        f"{tuple(logits.shape)}"
    )

    expected_output = (
        test_batch_size,
        config.NUM_CLASSES
    )

    if tuple(logits.shape) != expected_output:

        raise RuntimeError(
            f"Unexpected output shape.\n"
            f"Expected: {expected_output}\n"
            f"Got:      {tuple(logits.shape)}"
        )

    print(
        "\n  ✓ Forward pass successful"
    )


# =====================================================================
# GRADIENT FLOW TEST
# =====================================================================

def test_gradient_flow(
    model,
    device
):

    print(
        "\n" + "=" * 60
    )

    print(
        "CHECKING GRADIENT FLOW"
    )

    print(
        "=" * 60
    )

    test_batch_size = min(
        getattr(config, "BATCH_SIZE", 8),
        8
    )

    images = torch.randn(
        test_batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device,
        requires_grad=False
    )

    handcrafted_features = torch.randn(
        test_batch_size,
        config.HANDCRAFTED_FEATURE_DIM,
        device=device,
        requires_grad=False
    )

    labels = torch.randint(
        low=0,
        high=config.NUM_CLASSES,
        size=(test_batch_size,),
        device=device
    )

    model.train()

    model.zero_grad(
        set_to_none=True
    )

    logits = model(
        images,
        handcrafted_features
    )

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels
    )

    loss.backward()

    # ------------------------------------------------------------
    # Check backbone gradients
    # ------------------------------------------------------------

    backbone_gradients = any(
        parameter.grad is not None
        and torch.isfinite(
            parameter.grad
        ).all()
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    )

    # ------------------------------------------------------------
    # Check handcrafted branch gradients
    # ------------------------------------------------------------

    handcrafted_gradients = any(
        parameter.grad is not None
        and torch.isfinite(
            parameter.grad
        ).all()
        for parameter in model.handcrafted_branch.parameters()
        if parameter.requires_grad
    )

    # ------------------------------------------------------------
    # Check classifier gradients
    # ------------------------------------------------------------

    classifier_gradients = any(
        parameter.grad is not None
        and torch.isfinite(
            parameter.grad
        ).all()
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    )

    print(
        f"  Backbone gradients      : "
        f"{backbone_gradients}"
    )

    print(
        f"  Handcrafted gradients   : "
        f"{handcrafted_gradients}"
    )

    print(
        f"  Classifier gradients    : "
        f"{classifier_gradients}"
    )

    if not backbone_gradients:

        raise RuntimeError(
            "No valid gradients detected in "
            "ConvNeXt-Tiny backbone."
        )

    if not handcrafted_gradients:

        raise RuntimeError(
            "No valid gradients detected in "
            "handcrafted feature branch."
        )

    if not classifier_gradients:

        raise RuntimeError(
            "No valid gradients detected in "
            "classification head."
        )

    print(
        "\n  ✓ Gradient flow verified"
    )


# =====================================================================
# DEEP EMBEDDING TEST
# =====================================================================

def test_embedding_extraction(
    model,
    device
):

    print(
        "\n" + "=" * 60
    )

    print(
        "CHECKING DEEP EMBEDDING"
    )

    print(
        "=" * 60
    )

    test_batch_size = min(
        getattr(config, "BATCH_SIZE", 8),
        8
    )

    images = torch.randn(
        test_batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device
    )

    model.eval()

    with torch.no_grad():

        embedding = (
            model.extract_deep_embedding(
                images
            )
        )

    print(
        f"  Embedding shape: "
        f"{tuple(embedding.shape)}"
    )

    expected_shape = (
        test_batch_size,
        CONVNEXT_TINY_FEATURE_DIM
    )

    if tuple(embedding.shape) != expected_shape:

        raise RuntimeError(
            f"Unexpected embedding shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {tuple(embedding.shape)}"
        )

    if not torch.isfinite(
        embedding
    ).all():

        raise RuntimeError(
            "NaN or Inf detected in deep embedding."
        )

    print(
        "\n  ✓ 768-D deep embedding verified"
    )


# =====================================================================
# FUSED FEATURE TEST
# =====================================================================

def test_fused_features(
    model,
    device
):

    print(
        "\n" + "=" * 60
    )

    print(
        "CHECKING FEATURE FUSION"
    )

    print(
        "=" * 60
    )

    test_batch_size = min(
        getattr(config, "BATCH_SIZE", 8),
        8
    )

    images = torch.randn(
        test_batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device
    )

    handcrafted_features = torch.randn(
        test_batch_size,
        config.HANDCRAFTED_FEATURE_DIM,
        device=device
    )

    model.eval()

    with torch.no_grad():

        fused = model.extract_fused_features(
            images,
            handcrafted_features
        )

    print(
        f"  Deep features       : "
        f"{CONVNEXT_TINY_FEATURE_DIM}"
    )

    print(
        f"  Handcrafted branch  : 256"
    )

    print(
        f"  Fused features      : "
        f"{tuple(fused.shape)}"
    )

    expected_dim = (
        CONVNEXT_TINY_FEATURE_DIM
        +
        256
    )

    expected_shape = (
        test_batch_size,
        expected_dim
    )

    if tuple(fused.shape) != expected_shape:

        raise RuntimeError(
            f"Unexpected fused feature shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {tuple(fused.shape)}"
        )

    print(
        "\n  ✓ Feature fusion verified"
    )


# =====================================================================
# MAIN SELF-TEST
# =====================================================================

def main():

    start_time = time.perf_counter()

    print(
        "=" * 60
    )

    print(
        "PHASE 2 - STEP 7A: "
        "CONVNEXT-TINY MODEL SELF-TEST"
    )

    print(
        "=" * 60
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

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
            torch.cuda.get_device_properties(
                0
            ).total_memory
            /
            (1024 ** 3)
        )

        print(
            f"GPU VRAM: "
            f"{total_memory:.2f} GB"
        )

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    print(
        "\nModel configuration:"
    )

    print(
        f"  Architecture        : ConvNeXt-Tiny"
    )

    print(
        f"  Image size          : "
        f"{config.IMAGE_SIZE}x{config.IMAGE_SIZE}"
    )

    print(
        f"  Deep embedding      : "
        f"{CONVNEXT_TINY_FEATURE_DIM}"
    )

    print(
        f"  Handcrafted input   : "
        f"{config.HANDCRAFTED_FEATURE_DIM}"
    )

    print(
        f"  Handcrafted branch  : 256"
    )

    print(
        f"  Fusion dimension    : "
        f"{CONVNEXT_TINY_FEATURE_DIM + 256}"
    )

    print(
        f"  Number of classes   : "
        f"{config.NUM_CLASSES}"
    )

    print(
        f"  Pretrained          : True"
    )

    print(
        f"  Full fine-tuning    : True"
    )

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "BUILDING CONVNEXT-TINY HYBRID CLASSIFIER"
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
    ).to(DEVICE)

    print(
        "\nModel created successfully."
    )

    # ------------------------------------------------------------
    # Parameter information
    # ------------------------------------------------------------

    total_params, trainable_params = (
        count_parameters(model)
    )

    model_size_mb = estimate_model_size(
        model
    )

    print(
        "\nModel parameters:"
    )

    print(
        f"  Backbone : "
        f"{sum(p.numel() for p in model.backbone.parameters()):,}"
    )

    print(
        f"  Head     : "
        f"{sum(p.numel() for p in model.handcrafted_branch.parameters()) + sum(p.numel() for p in model.classifier.parameters()):,}"
    )

    print(
        f"  Total    : "
        f"{total_params:,}"
    )

    print(
        f"  Trainable: "
        f"{trainable_params:,}"
    )

    print(
        f"  Size     : "
        f"{model_size_mb:.2f} MB"
    )

    # ------------------------------------------------------------
    # Verify backbone is trainable
    # ------------------------------------------------------------

    backbone_trainable = (
        check_backbone_trainable(
            model
        )
    )

    print(
        f"\nBackbone trainable: "
        f"{backbone_trainable}"
    )

    # ------------------------------------------------------------
    # Forward test
    # ------------------------------------------------------------

    test_forward_pass(
        model,
        DEVICE
    )

    # ------------------------------------------------------------
    # Gradient test
    # ------------------------------------------------------------

    test_gradient_flow(
        model,
        DEVICE
    )

    # ------------------------------------------------------------
    # Embedding test
    # ------------------------------------------------------------

    test_embedding_extraction(
        model,
        DEVICE
    )

    # ------------------------------------------------------------
    # Fusion test
    # ------------------------------------------------------------

    test_fused_features(
        model,
        DEVICE
    )

    # ------------------------------------------------------------
    # GPU memory information
    # ------------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

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

    # ------------------------------------------------------------
    # Final
    # ------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 2 - STEP 7A COMPLETE ✓"
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
        "  ✓ ConvNeXt-Tiny"
    )

    print(
        "  ✓ ImageNet pretrained weights"
    )

    print(
        "  ✓ Full backbone fine-tuning"
    )

    print(
        "  ✓ 768-D deep embedding"
    )

    print(
        "  ✓ 1320-D handcrafted features"
    )

    print(
        "  ✓ Handcrafted feature branch"
    )

    print(
        "  ✓ Deep branch normalization"
    )

    print(
        "  ✓ Handcrafted branch normalization"
    )

    print(
        "  ✓ Feature fusion"
    )

    print(
        "  ✓ Classification head"
    )

    print(
        "  ✓ Forward pass"
    )

    print(
        "  ✓ Backbone gradient flow"
    )

    print(
        "  ✓ Handcrafted branch gradient flow"
    )

    print(
        "  ✓ Classification head gradient flow"
    )

    print(
        "  ✓ Deep embedding extraction"
    )

    print(
        "  ✓ Fused feature extraction"
    )

    print(
        "\nNext:"
    )

    print(
        "  python src\\08A_train_centralized_convnext.py"
    )

    print(
        "\nMobileNetV3-Small baseline remains:"
    )

    print(
        "  07_model.py"
    )

    print(
        "  08_train_centralized.py"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()