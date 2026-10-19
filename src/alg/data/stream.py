"""
Continual Stream Simulator and Sample Data Structures.

Simulates ordered data streams with controlled perturbations:
- Clean samples
- Redundant / duplicate samples
- Symmetrically noisy labels
- Perturbed features (Gaussian, cutout, blur)
- Distribution shifts across stream segments

Strict Fairness Guarantee:
- The `StreamSample.training_tuple()` method returns ONLY (image, observed_label).
- All perturbation flags and ground-truth labels are quarantined inside
  `SampleMetadata` for post-hoc evaluation and analysis only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from alg.data.perturbations import apply_feature_corruption, apply_label_noise

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleMetadata:
    """
    Debug and evaluation-only ground truth metadata.

    CRITICAL: Never expose these fields to the model or gate during training.
    """

    true_label: int
    is_noisy_label: bool
    is_corrupted_feature: bool
    is_redundant: bool
    segment_id: int
    original_dataset_index: int
    corruption_type: str = "none"


@dataclass
class StreamSample:
    """
    Represents an individual sample arriving sequentially in the continual stream.
    """

    sample_id: str
    stream_index: int
    image: torch.Tensor
    observed_label: int
    metadata: SampleMetadata

    def training_tuple(self) -> tuple[torch.Tensor, int]:
        """
        Extract training-safe input.

        Guarantees that the learner receives only (image, observed_label)
        with zero access to ground truth or synthetic noise flags.
        """
        return self.image, self.observed_label


class StreamDataset(Dataset):
    """
    PyTorch Dataset wrapper around a list of StreamSamples.

    Yields `(image, observed_label)` by default for standard DataLoader batching.
    """

    def __init__(self, samples: Sequence[StreamSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.samples[idx].training_tuple()

    def get_sample(self, idx: int) -> StreamSample:
        """Access full sample including metadata (for evaluation only)."""
        return self.samples[idx]


class StreamSimulator:
    """
    Constructs reproducible, perturbation-injected data streams.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        stream_size: int = 10_000,
        label_noise_rate: float = 0.0,
        feature_corruption_rate: float = 0.0,
        redundancy_rate: float = 0.0,
        shift_type: str = "none",
        num_segments: int = 1,
        seed: int = 42,
        num_classes: int = 10,
    ) -> None:
        self.base_dataset = base_dataset
        self.stream_size = stream_size
        self.label_noise_rate = label_noise_rate
        self.feature_corruption_rate = feature_corruption_rate
        self.redundancy_rate = redundancy_rate
        self.shift_type = shift_type.lower()
        self.num_segments = max(1, num_segments)
        self.seed = seed
        self.num_classes = num_classes

        # Deterministic random generator
        self.rng = np.random.default_rng(seed)

        # Pre-build samples
        self.samples: list[StreamSample] = []
        self._build_stream()

    def _build_stream(self) -> None:
        """Construct the sequence of StreamSamples deterministically."""
        dataset_len = len(self.base_dataset)  # type: ignore[arg-type]
        segment_size = self.stream_size // self.num_segments

        # Index available dataset items by class for distribution shifting
        class_to_indices: dict[int, list[int]] = {c: [] for c in range(self.num_classes)}
        for idx in range(min(dataset_len, 50000)):
            _, target = self.base_dataset[idx]
            if isinstance(target, torch.Tensor):
                target = int(target.item())
            class_to_indices[int(target)].append(idx)

        all_indices = list(range(dataset_len))
        recent_buffer: list[StreamSample] = []
        buffer_size = 50

        for stream_idx in range(self.stream_size):
            segment_id = min(stream_idx // segment_size, self.num_segments - 1)

            # Check if this sample should be a redundant duplicate
            if (
                self.redundancy_rate > 0.0
                and len(recent_buffer) > 0
                and self.rng.random() < self.redundancy_rate
            ):
                source_sample = self.rng.choice(recent_buffer)
                sample = StreamSample(
                    sample_id=f"sample_{stream_idx:06d}",
                    stream_index=stream_idx,
                    image=source_sample.image.clone(),
                    observed_label=source_sample.observed_label,
                    metadata=SampleMetadata(
                        true_label=source_sample.metadata.true_label,
                        is_noisy_label=source_sample.metadata.is_noisy_label,
                        is_corrupted_feature=source_sample.metadata.is_corrupted_feature,
                        is_redundant=True,
                        segment_id=segment_id,
                        original_dataset_index=source_sample.metadata.original_dataset_index,
                        corruption_type=source_sample.metadata.corruption_type,
                    ),
                )
                self.samples.append(sample)
                continue

            # Select sample index based on distribution shift
            if self.shift_type == "class_shift" and self.num_segments > 1:
                # E.g. 2 segments: segment 0 favors first half of classes, segment 1 favors second half
                classes_per_seg = self.num_classes // self.num_segments
                favored_start = segment_id * classes_per_seg
                favored_end = favored_start + classes_per_seg
                favored_classes = list(range(favored_start, min(self.num_classes, favored_end)))

                # 85% probability of sampling from favored classes
                if self.rng.random() < 0.85 and len(favored_classes) > 0:
                    chosen_class = int(self.rng.choice(favored_classes))
                    pool = class_to_indices[chosen_class]
                    dataset_idx = int(self.rng.choice(pool)) if pool else int(self.rng.choice(all_indices))
                else:
                    dataset_idx = int(self.rng.choice(all_indices))
            else:
                dataset_idx = int(self.rng.choice(all_indices))

            img, target = self.base_dataset[dataset_idx]
            if isinstance(target, torch.Tensor):
                target = int(target.item())
            true_label = int(target)

            # Apply label noise
            observed_label, is_noisy = apply_label_noise(
                true_label=true_label,
                num_classes=self.num_classes,
                noise_rate=self.label_noise_rate,
                rng=self.rng,
            )

            # Apply feature corruption
            corr_type = "none"
            is_corrupted = False
            image_tensor = img if isinstance(img, torch.Tensor) else torch.tensor(img)

            if self.feature_corruption_rate > 0.0:
                corr_choice = str(self.rng.choice(["gaussian", "cutout", "blur"]))
                image_tensor, is_corrupted, corr_type = apply_feature_corruption(
                    image=image_tensor,
                    corruption_rate=self.feature_corruption_rate,
                    corruption_type=corr_choice,
                    severity=0.6,
                    rng=self.rng,
                )

            metadata = SampleMetadata(
                true_label=true_label,
                is_noisy_label=is_noisy,
                is_corrupted_feature=is_corrupted,
                is_redundant=False,
                segment_id=segment_id,
                original_dataset_index=dataset_idx,
                corruption_type=corr_type,
            )

            sample = StreamSample(
                sample_id=f"sample_{stream_idx:06d}",
                stream_index=stream_idx,
                image=image_tensor,
                observed_label=observed_label,
                metadata=metadata,
            )
            self.samples.append(sample)

            # Maintain buffer for redundancy injection
            recent_buffer.append(sample)
            if len(recent_buffer) > buffer_size:
                recent_buffer.pop(0)

        logger.info(
            "Stream constructed: %d samples across %d segment(s).",
            len(self.samples),
            self.num_segments,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[StreamSample]:
        """Iterate through the stream sequentially."""
        return iter(self.samples)

    def get_dataloader(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        """Create a standard DataLoader over the stream in strictly sequential order."""
        dataset = StreamDataset(self.samples)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Sequential arrival preserves stream order!
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Compute summary statistics over the constructed stream."""
        total = len(self.samples)
        noisy = sum(1 for s in self.samples if s.metadata.is_noisy_label)
        corrupted = sum(1 for s in self.samples if s.metadata.is_corrupted_feature)
        redundant = sum(1 for s in self.samples if s.metadata.is_redundant)
        clean = sum(
            1
            for s in self.samples
            if not s.metadata.is_noisy_label
            and not s.metadata.is_corrupted_feature
            and not s.metadata.is_redundant
        )

        segments = {}
        for s in self.samples:
            seg = s.metadata.segment_id
            segments[seg] = segments.get(seg, 0) + 1

        return {
            "total_samples": total,
            "clean_samples": clean,
            "clean_percentage": round(clean / total * 100, 2) if total else 0,
            "noisy_label_samples": noisy,
            "noisy_label_percentage": round(noisy / total * 100, 2) if total else 0,
            "corrupted_feature_samples": corrupted,
            "corrupted_feature_percentage": round(corrupted / total * 100, 2) if total else 0,
            "redundant_samples": redundant,
            "redundant_percentage": round(redundant / total * 100, 2) if total else 0,
            "segments": segments,
            "seed": self.seed,
            "shift_type": self.shift_type,
        }

    def export_manifest(self, filepath: str | Path) -> None:
        """
        Export stream manifest JSON file.

        Enables fair multi-baseline comparisons on identical stream sequences.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "summary": self.get_statistics(),
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "stream_index": s.stream_index,
                    "true_label": s.metadata.true_label,
                    "observed_label": s.observed_label,
                    "is_noisy_label": s.metadata.is_noisy_label,
                    "is_corrupted_feature": s.metadata.is_corrupted_feature,
                    "is_redundant": s.metadata.is_redundant,
                    "segment_id": s.metadata.segment_id,
                    "corruption_type": s.metadata.corruption_type,
                }
                for s in self.samples
            ],
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Stream manifest saved to '%s'.", path)
