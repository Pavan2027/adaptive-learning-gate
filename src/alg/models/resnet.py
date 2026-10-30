"""
Pre-trained ResNet-18 backbone for the Adaptive Learning Gate.

Wraps ``torchvision.models.resnet18`` with the standard ALG model interface:
- ``forward_features(x)``  → penultimate 512-D feature vector  (used by novelty / gate)
- ``forward(x)``           → unnormalized class logits
- ``self.feature_dim``     → 512 (read by PrototypeMemory)
- ``self.num_classes``     → number of target classes

The pre-trained ImageNet weights give a strong visual feature extractor out
of the box. Only the classification head (``self.classifier``) is randomly
initialized and trained from scratch on the target dataset.

Usage::

    from alg.models.resnet import ResNet18Backbone

    model = ResNet18Backbone(num_classes=100, pretrained=True)
    features = model.forward_features(x)   # (B, 512)
    logits   = model(x)                    # (B, 100)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Backbone(nn.Module):
    """
    Pre-trained ResNet-18 backbone with a trainable classification head.

    Architecture:
        ResNet-18 conv body (layers 1-4 + avgpool) → 512-D feature vector
        Linear(512 → num_classes) [Logits]

    The backbone weights can be frozen for fast linear-probe training or left
    unfrozen for full fine-tuning (default).
    """

    feature_dim: int = 512

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """
        Args:
            num_classes: Number of target output classes.
            pretrained: Load ImageNet pre-trained weights (default: True).
            freeze_backbone: Freeze all backbone parameters so only the
                classification head is trained (useful for linear probing).
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 512  # ResNet-18 avgpool output dimension

        # Load backbone with or without pre-trained weights
        if pretrained:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet18(weights=None)

        # Strip the original FC (it was trained for 1000 ImageNet classes)
        # We keep everything up to and including avgpool.
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool  # AdaptiveAvgPool2d((1, 1))

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

        # Fresh classification head — always trainable
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        # Ensure head is trainable even if backbone was frozen above
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate feature representations.

        Runs the full ResNet-18 conv body and average pooling, then flattens
        to a 512-D vector per sample.

        Args:
            x: Input tensor of shape (B, C, H, W).
               Supports any spatial resolution ≥ 8×8 thanks to adaptive pooling.

        Returns:
            Feature embedding tensor of shape (B, 512).
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)            # (B, 512, 1, 1)
        out = torch.flatten(out, 1)        # (B, 512)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass returning unnormalized class logits.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits

