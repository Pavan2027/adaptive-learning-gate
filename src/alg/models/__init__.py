"""
Model architectures and factory for the Adaptive Learning Gate.

Exports:
- CompactCNN: Standard 4-layer CNN with penultimate feature extraction.
- ModularMLP: Configurable multi-layer perceptron.
- build_model: Factory function instantiating architectures based on config.
"""

from __future__ import annotations

import torch.nn as nn

from alg.models.cnn import CompactCNN
from alg.models.mlp import ModularMLP


def build_model(
    architecture: str = "cnn",
    num_classes: int = 10,
    in_channels: int = 3,
    image_size: tuple[int, int] = (32, 32),
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Factory function to instantiate models.

    Args:
        architecture: 'cnn' or 'mlp'.
        num_classes: Number of target classification categories.
        in_channels: Number of input image channels (1 for grayscale, 3 for RGB).
        image_size: Tuple (height, width) of input images.
        hidden_dims: List of hidden dimensions for MLP architecture.
        dropout: Dropout probability.

    Returns:
        Instantiated nn.Module ready for training.
    """
    arch = architecture.lower()
    if arch == "cnn":
        return CompactCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif arch == "mlp":
        input_dim = in_channels * image_size[0] * image_size[1]
        return ModularMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Supported architectures: ['cnn', 'mlp']."
        )


__all__ = ["CompactCNN", "ModularMLP", "build_model"]
