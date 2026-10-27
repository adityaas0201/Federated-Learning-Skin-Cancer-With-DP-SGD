"""
07_model.py
------------
PHASE 2 - STEP 7

Hybrid model:
    MobileNetV3-Small deep features
                +
    1320 handcrafted features
                |
                v
        Normalized feature fusion
                |
                v
        Classification head
"""

import os
import sys

import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs"
    )
)

import config


# =====================================================================
# HYBRID CLASSIFIER
# =====================================================================

class HybridClassifier(nn.Module):

    def __init__(
        self,
        num_classes=None,
        handcrafted_feature_dim=None,
        pretrained=True,
        head_hidden_dim=512,
        dropout=0.30,
    ):
        super().__init__()

        num_classes = num_classes or config.NUM_CLASSES
        handcrafted_feature_dim = (
            handcrafted_feature_dim
            or config.HANDCRAFTED_FEATURE_DIM
        )

        # -------------------------------------------------------------
        # MobileNetV3-Small
        # -------------------------------------------------------------

        weights = (
            MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        backbone = mobilenet_v3_small(weights=weights)

        self.feature_extractor = backbone.features
        self.avgpool = backbone.avgpool

        # MobileNetV3-Small embedding = 576
        self.deep_embedding_dim = backbone.classifier[0].in_features

        self.handcrafted_feature_dim = handcrafted_feature_dim

        # -------------------------------------------------------------
        # Normalize each branch before fusion
        # -------------------------------------------------------------

        self.deep_norm = nn.LayerNorm(
            self.deep_embedding_dim
        )

        self.handcrafted_norm = nn.LayerNorm(
            handcrafted_feature_dim
        )

        # -------------------------------------------------------------
        # Feature fusion
        # -------------------------------------------------------------

        self.fused_dim = (
            self.deep_embedding_dim
            + handcrafted_feature_dim
        )

        # -------------------------------------------------------------
        # Classification head
        # -------------------------------------------------------------

        self.classification_head = nn.Sequential(

            nn.Linear(
                self.fused_dim,
                head_hidden_dim
            ),

            nn.BatchNorm1d(
                head_hidden_dim
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                head_hidden_dim,
                256
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                256,
                num_classes
            ),
        )

    # =================================================================
    # DEEP EMBEDDING
    # =================================================================

    def extract_deep_embedding(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:

        x = self.feature_extractor(images)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        return x

    # =================================================================
    # FORWARD
    # =================================================================

    def forward(
        self,
        images: torch.Tensor,
        handcrafted_features: torch.Tensor
    ) -> torch.Tensor:

        # -------------------------------------------------------------
        # Deep branch
        # -------------------------------------------------------------

        deep_embedding = self.extract_deep_embedding(
            images
        )

        deep_embedding = self.deep_norm(
            deep_embedding
        )

        # -------------------------------------------------------------
        # Handcrafted branch
        # -------------------------------------------------------------

        handcrafted_features = self.handcrafted_norm(
            handcrafted_features
        )

        # -------------------------------------------------------------
        # Fusion
        # -------------------------------------------------------------

        fused = torch.cat(
            [
                deep_embedding,
                handcrafted_features
            ],
            dim=1
        )

        # -------------------------------------------------------------
        # Classification
        # -------------------------------------------------------------

        logits = self.classification_head(
            fused
        )

        return logits

    # =================================================================
    # PARAMETER INFORMATION
    # =================================================================

    def count_parameters(self):

        backbone_params = sum(
            p.numel()
            for p in self.feature_extractor.parameters()
        )

        head_params = sum(
            p.numel()
            for p in self.classification_head.parameters()
        )

        total_params = sum(
            p.numel()
            for p in self.parameters()
        )

        return {
            "backbone_params": backbone_params,
            "head_params": head_params,
            "total_params": total_params,
            "total_params_millions": round(
                total_params / 1e6,
                3
            ),
            "approx_size_mb": round(
                total_params * 4 / (1024 ** 2),
                2
            ),
        }


# =====================================================================
# SELF TEST
# =====================================================================

def main():

    print("=" * 60)
    print("PHASE 2 - STEP 7: MODEL SELF-TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------

    print("\nBuilding HybridClassifier...")

    model = HybridClassifier(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=config.HANDCRAFTED_FEATURE_DIM,
        pretrained=True,
    ).to(device)

    # -------------------------------------------------------------
    # Parameter count
    # -------------------------------------------------------------

    info = model.count_parameters()

    print("\nModel parameters:")
    print(
        f"  Backbone : {info['backbone_params']:,}"
    )
    print(
        f"  Head     : {info['head_params']:,}"
    )
    print(
        f"  Total    : {info['total_params']:,}"
    )
    print(
        f"  Size     : {info['approx_size_mb']} MB"
    )

    # -------------------------------------------------------------
    # Dummy input
    # -------------------------------------------------------------

    batch_size = 16

    dummy_images = torch.randn(
        batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device
    )

    dummy_features = torch.randn(
        batch_size,
        config.HANDCRAFTED_FEATURE_DIM,
        device=device
    )

    # -------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------

    print("\nRunning forward pass...")

    model.train()

    logits = model(
        dummy_images,
        dummy_features
    )

    print(
        f"  Images   : {tuple(dummy_images.shape)}"
    )

    print(
        f"  Features : {tuple(dummy_features.shape)}"
    )

    print(
        f"  Output   : {tuple(logits.shape)}"
    )

    expected_shape = (
        batch_size,
        config.NUM_CLASSES
    )

    assert logits.shape == expected_shape

    # -------------------------------------------------------------
    # Gradient test
    # -------------------------------------------------------------

    print("\nChecking gradient flow...")

    labels = torch.randint(
        0,
        config.NUM_CLASSES,
        (batch_size,),
        device=device
    )

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels
    )

    loss.backward()

    backbone_has_grad = any(
        p.grad is not None
        and p.grad.abs().sum().item() > 0
        for p in model.feature_extractor.parameters()
    )

    head_has_grad = any(
        p.grad is not None
        and p.grad.abs().sum().item() > 0
        for p in model.classification_head.parameters()
    )

    print(
        f"  Backbone gradients : {backbone_has_grad}"
    )

    print(
        f"  Head gradients     : {head_has_grad}"
    )

    assert backbone_has_grad
    assert head_has_grad

    # -------------------------------------------------------------
    # Embedding test
    # -------------------------------------------------------------

    print("\nChecking deep embedding...")

    model.eval()

    with torch.no_grad():

        embedding = model.extract_deep_embedding(
            dummy_images
        )

    print(
        f"  Embedding shape: {tuple(embedding.shape)}"
    )

    assert embedding.shape == (
        batch_size,
        model.deep_embedding_dim
    )

    # -------------------------------------------------------------
    # Complete
    # -------------------------------------------------------------

    print("\n" + "=" * 60)
    print("PHASE 2 - STEP 7 COMPLETE ✓")
    print("=" * 60)

    print(
        "\nVerified:"
    )

    print(
        "  ✓ MobileNetV3-Small"
    )

    print(
        "  ✓ Full backbone fine-tuning"
    )

    print(
        "  ✓ 576-D deep embedding"
    )

    print(
        "  ✓ 1320-D handcrafted features"
    )

    print(
        "  ✓ Branch normalization"
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
        "  ✓ Gradient flow"
    )

    print(
        "  ✓ Embedding extraction"
    )

    print(
        "\nNext:"
    )

    print(
        "  python src\\08_train_centralized.py"
    )


if __name__ == "__main__":
    main()