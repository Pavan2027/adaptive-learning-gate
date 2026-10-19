"""
Tests for alg.data.datasets.

Covers:
- Dataset metadata lookup for CIFAR-10, Fashion-MNIST, and MNIST.
- Unsupported dataset exception handling.
- Train/Validation deterministic split separation and non-overlap.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from alg.data.datasets import (
    DATASET_REGISTRY,
    DatasetMetadata,
    get_dataset_metadata,
)


class TestDatasetMetadata:
    def test_cifar10_metadata(self) -> None:
        meta = get_dataset_metadata("cifar10")
        assert meta.num_classes == 10
        assert meta.in_channels == 3
        assert meta.image_size == (32, 32)
        assert len(meta.mean) == 3

    def test_fashion_mnist_metadata(self) -> None:
        meta = get_dataset_metadata("fashion_mnist")
        assert meta.num_classes == 10
        assert meta.in_channels == 1
        assert meta.image_size == (28, 28)

    def test_mnist_metadata(self) -> None:
        meta = get_dataset_metadata("mnist")
        assert meta.num_classes == 10
        assert meta.in_channels == 1

    def test_case_and_hyphen_insensitive(self) -> None:
        meta1 = get_dataset_metadata("Fashion-MNIST")
        meta2 = get_dataset_metadata("fashion_mnist")
        assert meta1 == meta2

    def test_unknown_dataset_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dataset"):
            get_dataset_metadata("imagenet_1k")


class TestDataSplitting:
    def test_split_separation_no_data_leakage(self) -> None:
        """Verify that train and validation subsets share zero sample indices."""
        total_samples = 100
        val_fraction = 0.2
        num_val = int(total_samples * val_fraction)
        num_train = total_samples - num_val

        dataset = TensorDataset(torch.randn(total_samples, 3, 32, 32), torch.randint(0, 10, (total_samples,)))
        generator = torch.Generator().manual_seed(42)
        train_sub, val_sub = random_split(dataset, [num_train, num_val], generator=generator)

        train_indices = set(train_sub.indices)
        val_indices = set(val_sub.indices)

        # No intersection between train and validation indices
        assert train_indices.isdisjoint(val_indices)
        # Total size matches
        assert len(train_indices) + len(val_indices) == total_samples
