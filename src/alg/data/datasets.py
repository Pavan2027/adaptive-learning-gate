"""
Dataset loading and train/val/test splitting for the Adaptive Learning Gate.

Supports:
- CIFAR-10      : 10 classes, 3 channels (RGB), 32x32 (Primary research benchmark)
- Fashion-MNIST : 10 classes, 1 channel (Grayscale), 28x28 (Prototyping benchmark)
- MNIST         : 10 classes, 1 channel (Grayscale), 28x28 (Sanity checking)

Data splitting policy:
- The standard training set is explicitly partitioned into train (e.g. 90%) and
  validation (e.g. 10%) subsets using a fixed random generator.
- The official test set is strictly held out and only evaluated at the end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata describing dataset structure and normalization constants."""

    name: str
    num_classes: int
    in_channels: int
    image_size: Tuple[int, int]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


DATASET_REGISTRY: dict[str, DatasetMetadata] = {
    "cifar10": DatasetMetadata(
        name="cifar10",
        num_classes=10,
        in_channels=3,
        image_size=(32, 32),
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    "fashion_mnist": DatasetMetadata(
        name="fashion_mnist",
        num_classes=10,
        in_channels=1,
        image_size=(28, 28),
        mean=(0.2860,),
        std=(0.3530,),
    ),
    "mnist": DatasetMetadata(
        name="mnist",
        num_classes=10,
        in_channels=1,
        image_size=(28, 28),
        mean=(0.1307,),
        std=(0.3081,),
    ),
}


def get_dataset_metadata(dataset_name: str) -> DatasetMetadata:
    """Retrieve metadata for a supported dataset."""
    name_clean = dataset_name.lower().replace("-", "_")
    if name_clean not in DATASET_REGISTRY:
        supported = list(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unsupported dataset: '{dataset_name}'. Supported datasets: {supported}"
        )
    return DATASET_REGISTRY[name_clean]


def _build_transforms(metadata: DatasetMetadata, is_train: bool = True) -> transforms.Compose:
    """
    Construct standardized transforms for train or evaluation.

    Basic data normalization is applied. Training transforms can optionally
    include standard data augmentation (e.g. random crop/horizontal flip for CIFAR-10).
    """
    transform_list = []
    if is_train and metadata.name == "cifar10":
        transform_list.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(metadata.mean, metadata.std),
        ]
    )
    return transforms.Compose(transform_list)


def load_raw_dataset(
    dataset_name: str,
    root: str | Path = "data/",
    train: bool = True,
    download: bool = True,
    transform: transforms.Compose | None = None,
) -> Dataset:
    """Load the underlying torchvision dataset."""
    name_clean = dataset_name.lower().replace("-", "_")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    if name_clean == "cifar10":
        dataset_dir = root_path / "cifar-10-batches-py"
        if download and not dataset_dir.exists():
            print(f"[INFO] Downloading CIFAR-10 archive (~170MB) to '{root_path}'... (one-time setup)")
        return datasets.CIFAR10(
            root=str(root_path), train=train, download=download, transform=transform
        )
    elif name_clean == "fashion_mnist":
        dataset_dir = root_path / "FashionMNIST"
        if download and not dataset_dir.exists():
            print(f"[INFO] Downloading Fashion-MNIST archive (~30MB) to '{root_path}'... (one-time setup)")
        return datasets.FashionMNIST(
            root=str(root_path), train=train, download=download, transform=transform
        )
    elif name_clean == "mnist":
        dataset_dir = root_path / "MNIST"
        if download and not dataset_dir.exists():
            print(f"[INFO] Downloading MNIST archive (~10MB) to '{root_path}'... (one-time setup)")
        return datasets.MNIST(
            root=str(root_path), train=train, download=download, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")


def get_dataloaders(
    dataset_name: str = "cifar10",
    root: str | Path = "data/",
    batch_size: int = 64,
    val_fraction: float = 0.1,
    seed: int = 42,
    download: bool = True,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    """
    Load dataset and construct reproducible Train, Validation, and Test DataLoaders.

    Args:
        dataset_name: 'cifar10', 'fashion_mnist', or 'mnist'.
        root: Storage directory for downloaded datasets.
        batch_size: Number of samples per batch.
        val_fraction: Proportion of training set reserved for validation (e.g. 0.1 = 10%).
        seed: Random seed for deterministic train/val splitting.
        download: Automatically download the dataset if missing.
        num_workers: Number of DataLoader worker subprocesses (0 = main process).
        pin_memory: Pin memory for faster GPU transfer. Default is True if CUDA is available.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, metadata).
    """
    if not (0.0 < val_fraction < 0.5):
        raise ValueError(
            f"val_fraction must be in (0.0, 0.5), got {val_fraction}."
        )

    metadata = get_dataset_metadata(dataset_name)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Load full training set with training transforms
    train_transform = _build_transforms(metadata, is_train=True)
    eval_transform = _build_transforms(metadata, is_train=False)

    full_train_dataset = load_raw_dataset(
        metadata.name, root=root, train=True, download=download, transform=train_transform
    )
    test_dataset = load_raw_dataset(
        metadata.name, root=root, train=False, download=download, transform=eval_transform
    )

    # Deterministic split into train and validation subsets
    num_total = len(full_train_dataset)  # type: ignore[arg-type]
    num_val = int(num_total * val_fraction)
    num_train = num_total - num_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_dataset, [num_train, num_val], generator=generator
    )

    logger.info(
        "Dataset '%s' loaded: %d train, %d val, %d test samples.",
        metadata.name,
        num_train,
        num_val,
        len(test_dataset),  # type: ignore[arg-type]
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, metadata
