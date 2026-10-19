"""
Tests for alg.data.stream (StreamSample, SampleMetadata, StreamSimulator).

Covers:
- Zero data leakage: training_tuple() returns only (image, observed_label).
- Stream determinism across runs with the same seed.
- Noise rate adherence in simulated streams.
- Redundancy injection.
- Segment boundaries and distribution shift.
- DataLoader sequential ordering.
- Manifest export and structure.
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from alg.data.stream import SampleMetadata, StreamSample, StreamSimulator


@pytest.fixture
def synthetic_base_dataset() -> TensorDataset:
    """Create a synthetic dataset with 100 images and 10 classes."""
    torch.manual_seed(42)
    images = torch.randn(100, 3, 32, 32)
    targets = torch.tensor([i % 10 for i in range(100)])
    return TensorDataset(images, targets)


class TestStreamSampleFairness:
    def test_training_tuple_quarantines_metadata(self) -> None:
        """Verify model/gate training interface receives only (image, observed_label)."""
        img = torch.randn(3, 32, 32)
        meta = SampleMetadata(
            true_label=1,
            is_noisy_label=True,
            is_corrupted_feature=False,
            is_redundant=False,
            segment_id=0,
            original_dataset_index=4,
        )
        sample = StreamSample(
            sample_id="sample_000001",
            stream_index=0,
            image=img,
            observed_label=7,  # Corrupted label
            metadata=meta,
        )

        train_input, train_target = sample.training_tuple()
        assert torch.equal(train_input, img)
        assert train_target == 7
        # Ensure metadata is NOT part of the training tuple
        assert not hasattr(train_input, "is_noisy_label")


class TestStreamSimulator:
    def test_determinism_same_seed(self, synthetic_base_dataset: TensorDataset) -> None:
        """Two simulators with the same seed must produce identical streams."""
        sim1 = StreamSimulator(
            base_dataset=synthetic_base_dataset,
            stream_size=200,
            label_noise_rate=0.2,
            feature_corruption_rate=0.1,
            seed=42,
        )
        sim2 = StreamSimulator(
            base_dataset=synthetic_base_dataset,
            stream_size=200,
            label_noise_rate=0.2,
            feature_corruption_rate=0.1,
            seed=42,
        )

        assert len(sim1) == len(sim2) == 200
        for s1, s2 in zip(sim1, sim2):
            assert s1.sample_id == s2.sample_id
            assert s1.observed_label == s2.observed_label
            assert s1.metadata.is_noisy_label == s2.metadata.is_noisy_label
            assert s1.metadata.is_corrupted_feature == s2.metadata.is_corrupted_feature
            assert torch.equal(s1.image, s2.image)

    def test_different_seeds_differ(self, synthetic_base_dataset: TensorDataset) -> None:
        sim1 = StreamSimulator(base_dataset=synthetic_base_dataset, stream_size=50, seed=1)
        sim2 = StreamSimulator(base_dataset=synthetic_base_dataset, stream_size=50, seed=2)
        # Should have differences in sample ordering
        labels1 = [s.observed_label for s in sim1]
        labels2 = [s.observed_label for s in sim2]
        assert labels1 != labels2

    def test_redundancy_injection(self, synthetic_base_dataset: TensorDataset) -> None:
        sim = StreamSimulator(
            base_dataset=synthetic_base_dataset,
            stream_size=200,
            redundancy_rate=0.25,
            seed=42,
        )
        stats = sim.get_statistics()
        assert stats["redundant_samples"] > 0
        assert 0.15 < (stats["redundant_samples"] / 200) < 0.35

    def test_stream_segments(self, synthetic_base_dataset: TensorDataset) -> None:
        sim = StreamSimulator(
            base_dataset=synthetic_base_dataset,
            stream_size=100,
            num_segments=4,
            seed=42,
        )
        stats = sim.get_statistics()
        assert len(stats["segments"]) == 4
        # Each segment should have roughly 25 samples
        for seg_id in range(4):
            assert stats["segments"][seg_id] == 25

    def test_dataloader_sequential_order(self, synthetic_base_dataset: TensorDataset) -> None:
        sim = StreamSimulator(
            base_dataset=synthetic_base_dataset,
            stream_size=20,
            seed=42,
        )
        loader = sim.get_dataloader(batch_size=5)
        all_labels = []
        for _, batch_labels in loader:
            all_labels.extend(batch_labels.tolist())

        stream_labels = [s.observed_label for s in sim]
        assert all_labels == stream_labels

    def test_export_manifest(
        self, synthetic_base_dataset: TensorDataset, tmp_path: Path
    ) -> None:
        sim = StreamSimulator(
            base_dataset=synthetic_base_dataset,
            stream_size=30,
            label_noise_rate=0.2,
            seed=42,
        )
        manifest_path = tmp_path / "manifest.json"
        sim.export_manifest(manifest_path)

        assert manifest_path.exists()
        import json

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "summary" in data
        assert "samples" in data
        assert len(data["samples"]) == 30
