"""
Tests for alg.config — configuration loading and validation.

Covers:
- Successful load of default.yaml
- Field overrides via Config() constructor
- Gate threshold ordering constraint (ignore < learn)
- Range validation on float fields
- Non-negative seed and positive batch_size constraints
- Type coercion / rejection of invalid types
- FileNotFoundError for missing config file
- Empty YAML file handling (should use all defaults)
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from alg.config import Config, DatasetConfig, GateConfig, OptimizerConfig, StreamConfig, load_config


# ── Load from file ────────────────────────────────────────────────────────────


class TestLoadDefaultConfig:
    def test_loads_without_error(self, default_config_path: Path) -> None:
        cfg = load_config(default_config_path)
        assert cfg is not None

    def test_seed_is_set(self, default_config_path: Path) -> None:
        cfg = load_config(default_config_path)
        assert cfg.seed == 42

    def test_batch_size_is_set(self, default_config_path: Path) -> None:
        cfg = load_config(default_config_path)
        assert cfg.batch_size == 128

    def test_gate_thresholds_ordered(self, default_config_path: Path) -> None:
        cfg = load_config(default_config_path)
        assert cfg.gate.ignore_threshold < cfg.gate.learn_threshold

    def test_dataset_name(self, default_config_path: Path) -> None:
        cfg = load_config(default_config_path)
        assert cfg.dataset.name == "cifar10"

    def test_optimizer_name(self, default_config_path: Path) -> None:
        cfg = load_config(default_config_path)
        assert cfg.optimizer.name == "adam"


class TestLoadConfigErrors:
    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config("this/path/does/not/exist.yaml")

    def test_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        """An empty YAML file should fall back to all Pydantic defaults."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        cfg = load_config(empty)
        assert cfg.seed == 42
        assert cfg.batch_size == 128


# ── Gate threshold validation ─────────────────────────────────────────────────


class TestGateThresholds:
    def test_valid_thresholds_accepted(self) -> None:
        gate = GateConfig(learn_threshold=0.7, ignore_threshold=0.2)
        assert gate.learn_threshold == 0.7
        assert gate.ignore_threshold == 0.2

    def test_ignore_greater_than_learn_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ignore_threshold"):
            GateConfig(learn_threshold=0.3, ignore_threshold=0.7)

    def test_equal_thresholds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GateConfig(learn_threshold=0.5, ignore_threshold=0.5)

    def test_threshold_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GateConfig(learn_threshold=0.6, ignore_threshold=-0.1)

    def test_threshold_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GateConfig(learn_threshold=1.5, ignore_threshold=0.3)

    def test_defer_queue_capacity_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GateConfig(
                learn_threshold=0.6,
                ignore_threshold=0.3,
                defer_queue_capacity=-1,
            )

    def test_defer_queue_capacity_zero_allowed(self) -> None:
        gate = GateConfig(
            learn_threshold=0.6,
            ignore_threshold=0.3,
            defer_queue_capacity=0,
        )
        assert gate.defer_queue_capacity == 0


# ── Seed and batch size ───────────────────────────────────────────────────────


class TestSeedAndBatchSize:
    def test_zero_seed_accepted(self) -> None:
        cfg = Config(seed=0)
        assert cfg.seed == 0

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Config(seed=-1)

    def test_positive_batch_size_accepted(self) -> None:
        cfg = Config(batch_size=128)
        assert cfg.batch_size == 128

    def test_zero_batch_size_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Config(batch_size=0)

    def test_string_seed_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Config(seed="not-a-number")  # type: ignore[arg-type]


# ── Optimizer validation ──────────────────────────────────────────────────────


class TestOptimizerConfig:
    def test_zero_lr_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OptimizerConfig(lr=0.0)

    def test_negative_lr_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OptimizerConfig(lr=-0.001)

    def test_negative_weight_decay_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OptimizerConfig(weight_decay=-1e-4)


# ── Stream config ─────────────────────────────────────────────────────────────


class TestStreamConfig:
    def test_zero_size_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamConfig(size=0)

    def test_noise_rate_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamConfig(label_noise_rate=1.1)

    def test_noise_rate_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamConfig(feature_corruption_rate=-0.1)


# ── Config override ───────────────────────────────────────────────────────────


class TestConfigOverride:
    def test_override_seed(self) -> None:
        cfg = Config(seed=99)
        assert cfg.seed == 99

    def test_nested_override(self) -> None:
        cfg = Config(gate={"learn_threshold": 0.8, "ignore_threshold": 0.2})  # type: ignore[arg-type]
        assert cfg.gate.learn_threshold == 0.8

    def test_model_dump_roundtrip(self, default_config_path: Path) -> None:
        """Config.model_dump() should produce a dict that re-creates an identical Config."""
        cfg = load_config(default_config_path)
        dump = cfg.model_dump()
        cfg2 = Config(**dump)
        assert cfg == cfg2
