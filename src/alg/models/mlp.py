"""
Multi-Layer Perceptron (MLP) architecture for lightweight image classification.

Suitable for baseline comparisons and fast CPU smoke-testing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModularMLP(nn.Module):
    """
    Modular Multi-Layer Perceptron.

    Flattens 2D/3D inputs and passes them through a sequence of fully-connected
    layers with BatchNorm, ReLU, and Dropout.
    """

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        num_classes: int = 10,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.feature_dim = hidden_dims[-1]

        layers: list[nn.Module] = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d, bias=False))
            layers.append(nn.BatchNorm1d(h_d))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_d = h_d

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate representations.

        Args:
            x: Input tensor of shape (B, ...).

        Returns:
            Embeddings of shape (B, feature_dim).
        """
        flat = torch.flatten(x, 1)
        return self.feature_extractor(flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits.

        Args:
            x: Input tensor of shape (B, ...).

        Returns:
            Logits of shape (B, num_classes).
        """
        features = self.forward_features(x)
        return self.classifier(features)
