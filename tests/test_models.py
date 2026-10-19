"""
Tests for alg.models (CompactCNN, ModularMLP, and build_model).

Covers:
- Forward pass output shape for CIFAR-10 (3x32x32) and Fashion-MNIST (1x28x28).
- Penultimate feature extraction shape via forward_features(x).
- ModularMLP forward and feature output shapes.
- build_model factory creation and error handling.
"""

import pytest
import torch

from alg.models import CompactCNN, ModularMLP, build_model


class TestCompactCNN:
    def test_cifar10_forward_shape(self) -> None:
        model = CompactCNN(in_channels=3, num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_grayscale_forward_shape(self) -> None:
        model = CompactCNN(in_channels=1, num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_features_shape(self) -> None:
        model = CompactCNN(in_channels=3, num_classes=10, feature_dim=256)
        x = torch.randn(4, 3, 32, 32)
        feats = model.forward_features(x)
        assert feats.shape == (4, 256)


class TestModularMLP:
    def test_mlp_forward_shape(self) -> None:
        model = ModularMLP(input_dim=3 * 32 * 32, num_classes=10, hidden_dims=[128, 64])
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_mlp_forward_features_shape(self) -> None:
        model = ModularMLP(input_dim=3 * 32 * 32, num_classes=10, hidden_dims=[128, 64])
        x = torch.randn(4, 3, 32, 32)
        feats = model.forward_features(x)
        assert feats.shape == (4, 64)


class TestBuildModelFactory:
    def test_build_cnn(self) -> None:
        model = build_model("cnn", num_classes=10, in_channels=3)
        assert isinstance(model, CompactCNN)

    def test_build_mlp(self) -> None:
        model = build_model("mlp", num_classes=10, in_channels=1, image_size=(28, 28))
        assert isinstance(model, ModularMLP)

    def test_invalid_arch_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model("transformer_xl")
