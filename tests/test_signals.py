"""
Tests for alg.signals (Uncertainty, Novelty, Reliability, and SignalExtractor).

Covers:
- Entropy uncertainty normalization: uniform -> 1.0, one-hot -> 0.0.
- Margin uncertainty bounds.
- PrototypeMemory cosine distance, update dynamics, and uninitialized edge case.
- Feature integrity (finite check, variance check).
- Label consistency (agreement vs strong conflict).
- Unified SignalExtractor batched execution and bounded [0, 1] outputs.
"""

import pytest
import torch

from alg.models import CompactCNN
from alg.signals import (
    PrototypeMemory,
    SignalExtractor,
    compute_feature_integrity,
    compute_label_consistency,
    compute_margin_uncertainty,
    compute_predictive_entropy,
    compute_reliability,
)


class TestPredictiveUncertainty:
    def test_uniform_logits_maximum_entropy(self) -> None:
        """Uniform probability distribution must produce normalized entropy = 1.0."""
        # 4 classes, equal logits
        logits = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
        entropy = compute_predictive_entropy(logits)
        assert torch.isclose(entropy, torch.tensor([1.0]), atol=1e-4)

    def test_one_hot_logits_minimum_entropy(self) -> None:
        """Completely confident prediction must produce normalized entropy ≈ 0.0."""
        logits = torch.tensor([[50.0, -50.0, -50.0, -50.0]])
        entropy = compute_predictive_entropy(logits)
        assert torch.isclose(entropy, torch.tensor([0.0]), atol=1e-4)

    def test_intermediate_logits_bounded(self) -> None:
        logits = torch.randn(10, 10)
        entropy = compute_predictive_entropy(logits)
        assert (entropy >= 0.0).all()
        assert (entropy <= 1.0).all()

    def test_margin_uncertainty_bounds(self) -> None:
        # Tied top two classes -> margin uncertainty = 1.0
        tied_logits = torch.tensor([[10.0, 10.0, 0.0, 0.0]])
        assert torch.isclose(compute_margin_uncertainty(tied_logits), torch.tensor([1.0]), atol=1e-3)

        # Huge dominant class -> margin uncertainty ≈ 0.0
        dominant_logits = torch.tensor([[50.0, 0.0, 0.0, 0.0]])
        assert torch.isclose(compute_margin_uncertainty(dominant_logits), torch.tensor([0.0]), atol=1e-3)


class TestNoveltyPrototypeMemory:
    def test_uninitialized_memory_returns_one(self) -> None:
        memory = PrototypeMemory(num_classes=4, feature_dim=8)
        feats = torch.randn(3, 8)
        novelty = memory.compute_novelty(feats)
        # All novelties should be 1.0 when no prototypes exist
        assert torch.equal(novelty, torch.ones(3))

    def test_identical_embedding_zero_novelty(self) -> None:
        memory = PrototypeMemory(num_classes=2, feature_dim=4, momentum=0.0)
        proto = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        labels = torch.tensor([0])
        memory.update(proto, labels)

        # Test with the identical vector
        novelty = memory.compute_novelty(proto)
        # Cosine distance = (1 - 1)/2 = 0.0
        assert torch.isclose(novelty, torch.tensor([0.0]), atol=1e-4)

    def test_orthogonal_embedding_half_novelty(self) -> None:
        memory = PrototypeMemory(num_classes=2, feature_dim=4, momentum=0.0)
        proto = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        labels = torch.tensor([0])
        memory.update(proto, labels)

        # Test with orthogonal vector
        orthogonal = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        novelty = memory.compute_novelty(orthogonal)
        # Cosine distance = (1 - 0)/2 = 0.5
        assert torch.isclose(novelty, torch.tensor([0.5]), atol=1e-4)


class TestReliability:
    def test_non_finite_pixels_zero_reliability(self) -> None:
        corrupted_img = torch.randn(1, 3, 32, 32)
        corrupted_img[0, 0, 0, 0] = float("nan")
        r_feat = compute_feature_integrity(corrupted_img)
        assert r_feat.item() == 0.0

    def test_normal_image_high_integrity(self) -> None:
        clean_img = torch.randn(2, 3, 32, 32) * 0.5
        r_feat = compute_feature_integrity(clean_img)
        assert (r_feat > 0.8).all()

    def test_label_consistency_matching_vs_conflicting(self) -> None:
        # Confident logits for class 0
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])

        # Case A: Observed label matches top class
        r_match = compute_label_consistency(logits, torch.tensor([0]))
        assert torch.isclose(r_match, torch.tensor([1.0]), atol=1e-3)

        # Case B: Observed label is class 1 (conflicts with prediction)
        r_conflict = compute_label_consistency(logits, torch.tensor([1]))
        assert r_conflict.item() < 0.01

    def test_composite_reliability_bounded(self) -> None:
        images = torch.randn(4, 3, 32, 32)
        logits = torch.randn(4, 10)
        labels = torch.tensor([0, 1, 2, 3])
        r = compute_reliability(images, logits, labels)
        assert (r >= 0.0).all()
        assert (r <= 1.0).all()


class TestSignalExtractor:
    def test_extract_batch_returns_valid_scores(self) -> None:
        model = CompactCNN(in_channels=3, num_classes=4, feature_dim=32)
        memory = PrototypeMemory(num_classes=4, feature_dim=32)
        extractor = SignalExtractor(model=model, prototype_memory=memory, device="cpu")

        images = torch.randn(5, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3, 0])

        scores = extractor.extract_batch(images, labels)
        assert len(scores) == 5

        for sc in scores:
            assert 0.0 <= sc.uncertainty <= 1.0
            assert 0.0 <= sc.novelty <= 1.0
            assert 0.0 <= sc.reliability <= 1.0
            assert "feature_integrity" in sc.raw_metrics
            assert "label_consistency" in sc.raw_metrics
