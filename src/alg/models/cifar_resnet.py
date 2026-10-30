"""
Native CIFAR ResNet backbone (ResNet-56) for the Adaptive Learning Gate.

Specifically engineered for 32x32 spatial dimensions (unlike ImageNet ResNet-18
which has a 7x7 stride-2 conv1 and maxpool that over-downsamples small images).
Architecture:
- conv1: 3x3 conv, stride 1, padding 1, 16 filters (preserves 32x32)
- layer1: 9 basic blocks (16 filters, 32x32)
- layer2: 9 basic blocks (32 filters, 16x16)
- layer3: 9 basic blocks (64 filters, 8x8)
- avgpool: AdaptiveAvgPool2d((1, 1)) -> 64-D embedding
- classifier: Linear(64 -> num_classes)

Pre-trained Benchmark Accuracies:
- CIFAR-10: 94.26% test accuracy
- CIFAR-100: 72.61% test accuracy
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CifarResNet56Backbone(nn.Module):
    """
    Native CIFAR ResNet-56 backbone with penultimate 64-D feature extraction.
    """

    feature_dim: int = 64

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """
        Args:
            num_classes: Target classes (10 for CIFAR-10, 100 for CIFAR-100).
            pretrained: Load pre-trained weights from chenyaofo/pytorch-cifar-models.
            freeze_backbone: Freeze conv layers for linear probe adaptation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 64

        model_name = "cifar10_resnet56" if num_classes == 10 else "cifar100_resnet56"
        base = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            model_name,
            pretrained=pretrained,
        )

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.avgpool = base.avgpool
        self.classifier = base.fc

        if freeze_backbone:
            for p in [
                *self.conv1.parameters(),
                *self.bn1.parameters(),
                *self.layer1.parameters(),
                *self.layer2.parameters(),
                *self.layer3.parameters(),
            ]:
                p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate 64-D feature representations.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Embeddings of shape (B, 64).
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass returning unnormalized class logits.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Logits of shape (B, num_classes).
        """
        features = self.forward_features(x)
        return self.classifier(features)

