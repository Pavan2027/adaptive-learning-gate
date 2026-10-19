"""
Convolutional Neural Network (CNN) architecture for image classification benchmarks.

Suitable for:
- CIFAR-10 (3x32x32)
- Fashion-MNIST (1x28x28)
- MNIST (1x28x28)

Includes:
- Feature extraction interface via `forward_features(x)` (essential for novelty signal in Phase 4).
- BatchNorm and Dropout for stability and regularization.
- Adaptive pooling for resolution invariance between 28x28 and 32x32 inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactCNN(nn.Module):
    """
    Standard 4-layer CNN for visual classification benchmarks.

    Architecture:
        Conv2d(in_channels -> 32) -> BN -> ReLU -> MaxPool2d
        Conv2d(32 -> 64)          -> BN -> ReLU -> MaxPool2d
        Conv2d(64 -> 128)         -> BN -> ReLU -> AdaptiveAvgPool2d((4, 4))
        Flatten -> Linear(2048 -> 256) -> BN -> ReLU -> Dropout(0.3) [Features]
        Linear(256 -> num_classes) [Logits]
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Penultimate projection layer (representations used for novelty/clustering)
        self.fc_feat = nn.Linear(128 * 4 * 4, feature_dim, bias=False)
        self.bn_feat = nn.BatchNorm1d(feature_dim)
        self.drop = nn.Dropout(p=dropout)

        # Output classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate feature representations.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature embedding tensor of shape (B, feature_dim).
        """
        out = self.pool(F.relu(self.bn1(self.conv1(x))))
        out = self.pool(F.relu(self.bn2(self.conv2(out))))
        out = self.adaptive_pool(F.relu(self.bn3(self.conv3(out))))
        out = torch.flatten(out, 1)
        feat = self.drop(F.relu(self.bn_feat(self.fc_feat(out))))
        return feat

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
