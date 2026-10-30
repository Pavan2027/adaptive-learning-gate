"""
Model architectures and factory for the Adaptive Learning Gate.

Exports:
- CompactCNN: Standard 4-layer CNN with penultimate feature extraction.
- ModularMLP: Configurable multi-layer perceptron.
- ResNet18Backbone: Pre-trained ResNet-18 backbone with ALG interface.
- build_model: Factory function instantiating architectures based on config.
"""

from __future__ import annotations

import torch.nn as nn

from alg.models.cifar_resnet import CifarResNet56Backbone
from alg.models.cnn import CompactCNN
from alg.models.mlp import ModularMLP
from alg.models.resnet import ResNet18Backbone


def build_model(
    architecture: str = "cnn",
    num_classes: int = 10,
    in_channels: int = 3,
    image_size: tuple[int, int] = (32, 32),
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory function to instantiate models.

    Args:
        architecture: 'cnn', 'mlp', 'resnet18', or 'resnet56'.
        num_classes: Number of target classification categories.
        in_channels: Number of input image channels (1 for grayscale, 3 for RGB).
        image_size: Tuple (height, width) of input images.
        hidden_dims: List of hidden dimensions for MLP architecture.
        dropout: Dropout probability.
        pretrained: Load pre-trained weights for ResNet-18/56 (default True).
        freeze_backbone: Freeze backbone parameters for linear probing.

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
    elif arch == "resnet18":
        return ResNet18Backbone(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    elif arch in ("resnet56", "cifar_resnet56", "cifar10_resnet56", "cifar100_resnet56"):
        return CifarResNet56Backbone(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Supported architectures: ['cnn', 'mlp', 'resnet18', 'resnet56']."
        )


__all__ = [
    "CompactCNN",
    "ModularMLP",
    "ResNet18Backbone",
    "CifarResNet56Backbone",
    "build_model",
]
