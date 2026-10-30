"""
Dataset loading and train/val/test splitting for the Adaptive Learning Gate.

Supports:
- CIFAR-10      : 10 classes,  3 channels (RGB),       32x32   (Primary benchmark)
- CIFAR-100     : 100 classes, 3 channels (RGB),       32x32   (Research benchmark)
- Tiny-ImageNet : 200 classes, 3 channels (RGB),       64x64   (Research benchmark)
- Fashion-MNIST : 10 classes,  1 channel  (Grayscale), 28x28   (Prototyping)
- MNIST         : 10 classes,  1 channel  (Grayscale), 28x28   (Sanity check)

Tiny-ImageNet notes:
  The dataset is not bundled in torchvision. Download it manually::

      wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
      unzip tiny-imagenet-200.zip -d data/

  This produces ``data/tiny-imagenet-200/{train,val,test}`` directories.
  The validation set is re-organised into per-class sub-directories automatically
  by the ``_prepare_tiny_imagenet_val`` helper on first use.

Data splitting policy:
- The standard training set is explicitly partitioned into train (e.g. 90%) and
  validation (e.g. 10%) subsets using a fixed random generator.
- The official test set is strictly held out and only evaluated at the end.
"""

from __future__ import annotations

import logging
import shutil
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
    "cifar100": DatasetMetadata(
        name="cifar100",
        num_classes=100,
        in_channels=3,
        image_size=(32, 32),
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
    "tiny_imagenet": DatasetMetadata(
        name="tiny_imagenet",
        num_classes=200,
        in_channels=3,
        image_size=(64, 64),
        mean=(0.4802, 0.4481, 0.3975),
        std=(0.2764, 0.2689, 0.2816),
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

    Basic data normalization is applied. Training transforms include standard
    data augmentation where applicable (random crop/flip for CIFAR and Tiny-ImageNet).
    """
    transform_list = []

    if is_train:
        if metadata.name in ("cifar10", "cifar100"):
            # 32×32 datasets: random crop with padding + horizontal flip
            transform_list.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        elif metadata.name == "tiny_imagenet":
            # 64×64 dataset: resize to 64 (in case val images differ), random crop + flip
            transform_list.extend(
                [
                    transforms.Resize(64),
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                ]
            )
    else:
        # Evaluation: only resize for Tiny-ImageNet to guarantee consistent size
        if metadata.name == "tiny_imagenet":
            transform_list.append(transforms.Resize(64))

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(metadata.mean, metadata.std),
        ]
    )
    return transforms.Compose(transform_list)


def _prepare_tiny_imagenet_val(root: Path) -> None:
    """
    Re-organise the Tiny-ImageNet validation set into per-class sub-directories.

    The official ``tiny-imagenet-200.zip`` puts all validation images flat in
    ``val/images/`` with a ``val/val_annotations.txt`` index.  torchvision's
    ``ImageFolder`` loader requires one sub-directory per class, so we create
    that structure on first call and leave a sentinel file so subsequent calls
    skip the work.

    Args:
        root: Path to ``tiny-imagenet-200/`` (parent of ``val/`` directory).
    """
    val_dir = root / "val"
    sentinel = val_dir / ".organised"
    if sentinel.exists():
        return  # Already done

    images_dir = val_dir / "images"
    annotations_file = val_dir / "val_annotations.txt"

    if not images_dir.exists() or not annotations_file.exists():
        # Validation set is already organised (e.g. from a different source) — skip
        return

    logger.info("Organising Tiny-ImageNet validation set into per-class folders...")

    # Read annotation file: each line is "filename\tclass_id\t..."
    img_to_class: dict[str, str] = {}
    with annotations_file.open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_to_class[parts[0]] = parts[1]

    # Move images into per-class sub-directories
    for img_name, class_id in img_to_class.items():
        src = images_dir / img_name
        dst_dir = val_dir / class_id
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / img_name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))

    sentinel.touch()
    logger.info("Tiny-ImageNet validation set organised at '%s'.", val_dir)


def load_raw_dataset(
    dataset_name: str,
    root: str | Path = "data/",
    train: bool = True,
    download: bool = True,
    transform: transforms.Compose | None = None,
) -> Dataset:
    """Load the underlying torchvision (or ImageFolder) dataset."""
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
    elif name_clean == "cifar100":
        dataset_dir = root_path / "cifar-100-python"
        if download and not dataset_dir.exists():
            print(f"[INFO] Downloading CIFAR-100 archive (~170MB) to '{root_path}'... (one-time setup)")
        return datasets.CIFAR100(
            root=str(root_path), train=train, download=download, transform=transform
        )
    elif name_clean == "tiny_imagenet":
        tiny_root = root_path / "tiny-imagenet-200"
        if not tiny_root.exists():
            raise FileNotFoundError(
                f"Tiny-ImageNet not found at '{tiny_root}'.\n"
                "Download it manually:\n"
                "  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
                f"  unzip tiny-imagenet-200.zip -d {root_path}\n"
                "Then re-run."
            )
        if train:
            split_dir = tiny_root / "train"
        else:
            # Organise val set into per-class folders on first use
            _prepare_tiny_imagenet_val(tiny_root)
            split_dir = tiny_root / "val"
        return datasets.ImageFolder(root=str(split_dir), transform=transform)
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
        dataset_name: 'cifar10', 'cifar100', 'tiny_imagenet', 'fashion_mnist', or 'mnist'.
        root: Storage directory for downloaded datasets.
        batch_size: Number of samples per batch.
        val_fraction: Proportion of training set reserved for validation (e.g. 0.1 = 10%).
        seed: Random seed for deterministic train/val splitting.
        download: Automatically download the dataset if missing (not supported for Tiny-ImageNet).
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
