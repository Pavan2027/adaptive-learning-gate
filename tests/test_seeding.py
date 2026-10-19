"""
Tests for alg.utils.seeding — seed utility.

Covers:
- set_seed() runs without error for valid seeds
- Negative seed raises ValueError
- Same seed → same Python random output (reproducibility)
- Different seeds → different Python random output
- NumPy reproducibility under same seed
- Torch seeding gracefully skipped when torch is absent (monkeypatched)
"""

import builtins
import random
from unittest.mock import patch

import numpy as np
import pytest

from alg.utils.seeding import set_seed


# ── Basic usage ───────────────────────────────────────────────────────────────


class TestSetSeedBasic:
    def test_runs_without_error_seed_zero(self) -> None:
        set_seed(0)

    def test_runs_without_error_seed_42(self) -> None:
        set_seed(42)

    def test_runs_without_error_large_seed(self) -> None:
        set_seed(2**31 - 1)

    def test_negative_seed_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            set_seed(-1)

    def test_negative_seed_large_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            set_seed(-100)


# ── Python random reproducibility ─────────────────────────────────────────────


class TestPythonRandomReproducibility:
    def test_same_seed_same_float(self) -> None:
        set_seed(42)
        val1 = random.random()
        set_seed(42)
        val2 = random.random()
        assert val1 == val2

    def test_different_seeds_different_float(self) -> None:
        set_seed(1)
        val1 = random.random()
        set_seed(2)
        val2 = random.random()
        assert val1 != val2

    def test_same_seed_same_sequence(self) -> None:
        set_seed(7)
        seq1 = [random.randint(0, 100) for _ in range(10)]
        set_seed(7)
        seq2 = [random.randint(0, 100) for _ in range(10)]
        assert seq1 == seq2


# ── NumPy reproducibility ─────────────────────────────────────────────────────


class TestNumpyReproducibility:
    def test_same_seed_same_array(self) -> None:
        set_seed(42)
        arr1 = np.random.rand(10)
        set_seed(42)
        arr2 = np.random.rand(10)
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_seeds_different_array(self) -> None:
        set_seed(10)
        arr1 = np.random.rand(5)
        set_seed(20)
        arr2 = np.random.rand(5)
        assert not np.array_equal(arr1, arr2)


# ── PyTorch absent — graceful degradation ─────────────────────────────────────


class TestTorchAbsent:
    def test_no_error_when_torch_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Simulate a missing torch installation.
        set_seed() should log a warning but NOT raise any exception.
        Python and NumPy seeds must still be applied.
        """
        real_import = builtins.__import__

        def _mock_import(name: str, *args, **kwargs):
            if name == "torch":
                raise ImportError("Simulated: torch not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        # Must not raise
        set_seed(42)

        # Python random seed must still have been applied
        val1 = random.random()
        set_seed(42)
        monkeypatch.setattr(builtins, "__import__", _mock_import)
        val2 = random.random()
        assert val1 == val2


# ── PyTorch present ───────────────────────────────────────────────────────────


class TestTorchPresent:
    def test_torch_seed_reproducible(self) -> None:
        """If torch is installed, same seed → same tensor."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed; skipping torch-specific seed test.")

        set_seed(42)
        t1 = torch.rand(5)
        set_seed(42)
        t2 = torch.rand(5)
        assert torch.equal(t1, t2), "Same seed should produce identical tensors."

    def test_different_torch_seeds_differ(self) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed.")

        set_seed(1)
        t1 = torch.rand(5)
        set_seed(2)
        t2 = torch.rand(5)
        assert not torch.equal(t1, t2)
