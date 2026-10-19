"""
Tests for alg.data.perturbations.

Covers:
- Label noise: zero rate, full rate, statistical distribution, class validity.
- Feature corruption: zero rate, Gaussian noise, cutout masking, blur.
- Invalid corruption type error handling.
"""

import numpy as np
import pytest
import torch

from alg.data.perturbations import apply_feature_corruption, apply_label_noise


class TestLabelNoise:
    def test_zero_rate_never_corrupts(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            label, is_corrupted = apply_label_noise(
                true_label=3, num_classes=10, noise_rate=0.0, rng=rng
            )
            assert label == 3
            assert not is_corrupted

    def test_full_rate_always_corrupts_to_different_valid_class(self) -> None:
        rng = np.random.default_rng(42)
        for true_lbl in range(10):
            for _ in range(20):
                label, is_corrupted = apply_label_noise(
                    true_label=true_lbl, num_classes=10, noise_rate=1.0, rng=rng
                )
                assert is_corrupted
                assert label != true_lbl
                assert 0 <= label < 10

    def test_statistical_corruption_rate(self) -> None:
        rng = np.random.default_rng(123)
        trials = 2000
        target_rate = 0.30
        corrupted_count = sum(
            apply_label_noise(true_label=0, num_classes=10, noise_rate=target_rate, rng=rng)[1]
            for _ in range(trials)
        )
        observed_rate = corrupted_count / trials
        # Should be within +/- 3% of target rate
        assert abs(observed_rate - target_rate) < 0.03


class TestFeatureCorruption:
    def test_zero_rate_leaves_image_unmodified(self) -> None:
        rng = np.random.default_rng(42)
        img = torch.randn(3, 32, 32)
        out, is_corrupted, ctype = apply_feature_corruption(
            img, corruption_rate=0.0, corruption_type="gaussian", rng=rng
        )
        assert not is_corrupted
        assert ctype == "none"
        assert torch.equal(img, out)

    def test_gaussian_noise_modifies_pixels(self) -> None:
        rng = np.random.default_rng(42)
        img = torch.zeros(3, 32, 32)
        out, is_corrupted, ctype = apply_feature_corruption(
            img, corruption_rate=1.0, corruption_type="gaussian", severity=0.5, rng=rng
        )
        assert is_corrupted
        assert ctype == "gaussian"
        assert out.shape == img.shape
        assert not torch.equal(img, out)

    def test_cutout_masks_patch_to_zero(self) -> None:
        rng = np.random.default_rng(42)
        img = torch.ones(3, 32, 32)
        out, is_corrupted, ctype = apply_feature_corruption(
            img, corruption_rate=1.0, corruption_type="cutout", severity=0.5, rng=rng
        )
        assert is_corrupted
        assert ctype == "cutout"
        assert out.shape == img.shape
        # Some pixels must now be 0.0
        assert (out == 0.0).any()

    def test_blur_preserves_shape(self) -> None:
        rng = np.random.default_rng(42)
        img = torch.randn(3, 32, 32)
        out, is_corrupted, ctype = apply_feature_corruption(
            img, corruption_rate=1.0, corruption_type="blur", rng=rng
        )
        assert is_corrupted
        assert ctype == "blur"
        assert out.shape == img.shape

    def test_invalid_corruption_type_raises_error(self) -> None:
        rng = np.random.default_rng(42)
        img = torch.randn(3, 32, 32)
        with pytest.raises(ValueError, match="Unknown corruption_type"):
            apply_feature_corruption(
                img, corruption_rate=1.0, corruption_type="invalid_filter", rng=rng
            )
