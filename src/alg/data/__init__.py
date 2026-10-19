"""
Data loading, stream construction, and perturbation utilities for the Adaptive Learning Gate.

Exports:
- get_dataloaders: Function to build train/val/test loaders with fixed splitting.
- get_dataset_metadata: Helper returning dataset metadata (channels, classes, size).
- DatasetMetadata: Dataclass containing dataset parameters.
- StreamSample: Typed sample arriving sequentially in the stream.
- SampleMetadata: Quarantined ground truth evaluation metadata.
- StreamSimulator: Engine constructing ordered, perturbation-injected streams.
- StreamDataset: PyTorch Dataset wrapper yielding (image, observed_label).
- apply_label_noise: Symmetric random label flipping.
- apply_feature_corruption: Controlled image perturbation.
"""

from alg.data.datasets import DatasetMetadata, get_dataloaders, get_dataset_metadata
from alg.data.perturbations import apply_feature_corruption, apply_label_noise
from alg.data.stream import (
    SampleMetadata,
    StreamDataset,
    StreamSample,
    StreamSimulator,
)

__all__ = [
    "get_dataloaders",
    "get_dataset_metadata",
    "DatasetMetadata",
    "StreamSample",
    "SampleMetadata",
    "StreamSimulator",
    "StreamDataset",
    "apply_label_noise",
    "apply_feature_corruption",
]
