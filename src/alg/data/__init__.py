"""
Data loading and stream construction for the Adaptive Learning Gate.

Exports:
- get_dataloaders: Function to build train/val/test loaders with fixed splitting.
- get_dataset_metadata: Helper returning dataset metadata (channels, classes, size).
- DatasetMetadata: Dataclass containing dataset parameters.
"""

from alg.data.datasets import DatasetMetadata, get_dataloaders, get_dataset_metadata

__all__ = ["get_dataloaders", "get_dataset_metadata", "DatasetMetadata"]
