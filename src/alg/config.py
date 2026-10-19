"""
Configuration loading and validation for the Adaptive Learning Gate.

Usage::

    from alg.config import load_config

    cfg = load_config("configs/default.yaml")
    print(cfg.seed)               # 42
    print(cfg.gate.learn_threshold)  # 0.6

All configuration fields are validated via Pydantic v2. Invalid values raise
``pydantic.ValidationError`` with a descriptive message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, model_validator

# ── Sub-config models ─────────────────────────────────────────────────────────


class DatasetConfig(BaseModel):
    """Dataset selection and storage settings."""

    name: str = Field(
        default="cifar10",
        description="Dataset name. Supported: 'cifar10', 'fashion_mnist', 'mnist'.",
    )
    root: str = Field(
        default="data/",
        description="Local root directory for dataset storage.",
    )
    download: bool = Field(
        default=True,
        description="Auto-download dataset if not found locally.",
    )


class ModelConfig(BaseModel):
    """Model architecture settings."""

    architecture: str = Field(
        default="mlp",
        description="Architecture type. Supported: 'mlp', 'cnn'. (Implemented Phase 2+)",
    )
    hidden_dims: list[int] = Field(
        default=[256, 128],
        description="Hidden layer sizes for MLP architecture.",
    )


class OptimizerConfig(BaseModel):
    """Optimizer hyperparameters."""

    name: str = Field(
        default="adam",
        description="Optimizer name. Supported: 'adam', 'sgd'.",
    )
    lr: float = Field(
        default=1e-3,
        gt=0.0,
        description="Learning rate. Must be > 0.",
    )
    weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        description="L2 weight decay regularization. Must be >= 0.",
    )


class StreamConfig(BaseModel):
    """Stream simulation settings (used Phase 3+)."""

    size: int = Field(
        default=10_000,
        ge=1,
        description="Number of samples in the training stream.",
    )
    shuffle: bool = Field(
        default=True,
        description="Shuffle the full dataset before constructing the stream.",
    )
    label_noise_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of stream labels to randomly corrupt. In [0.0, 1.0].",
    )
    feature_corruption_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of feature vectors to corrupt (e.g. Gaussian noise). In [0.0, 1.0].",
    )


class GateConfig(BaseModel):
    """Adaptive Learning Gate thresholds and defer queue settings."""

    learn_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Utility score threshold for LEARN action. "
            "utility >= learn_threshold → LEARN."
        ),
    )
    ignore_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Utility score threshold for IGNORE action. "
            "utility <= ignore_threshold → IGNORE. "
            "Must be strictly less than learn_threshold."
        ),
    )
    defer_queue_capacity: int = Field(
        default=500,
        ge=0,
        description=(
            "Maximum number of deferred samples the queue can hold. "
            "0 means unlimited (bounded only by memory)."
        ),
    )

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> "GateConfig":
        """Enforce ignore_threshold < learn_threshold."""
        if self.ignore_threshold >= self.learn_threshold:
            raise ValueError(
                f"Gate threshold constraint violated: "
                f"ignore_threshold ({self.ignore_threshold}) must be strictly less than "
                f"learn_threshold ({self.learn_threshold}). "
                f"The DEFER region requires a gap between the two thresholds."
            )
        return self


class OutputConfig(BaseModel):
    """Output path configuration."""

    log_dir: str = Field(default="logs/", description="Directory for training logs.")
    checkpoint_dir: str = Field(
        default="checkpoints/", description="Directory for model checkpoint files (.pt)."
    )
    results_dir: str = Field(
        default="results/", description="Directory for experiment result files (JSON, CSV)."
    )


# ── Top-level config ──────────────────────────────────────────────────────────


class Config(BaseModel):
    """
    Top-level project configuration for the Adaptive Learning Gate.

    Load from YAML using :func:`load_config`. All fields have safe defaults
    matching ``configs/default.yaml``.
    """

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    seed: Annotated[int, Field(ge=0, description="Global random seed. Must be >= 0.")] = 42
    batch_size: Annotated[
        int, Field(default=128, ge=1, description="Training batch size. Must be >= 1.")
    ] = 128
    epochs: Annotated[
        int, Field(default=5, ge=1, description="Number of baseline training epochs. Must be >= 1.")
    ] = 5
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# ── Loader ────────────────────────────────────────────────────────────────────


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """
    Load and validate project configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.
              Defaults to ``configs/default.yaml`` (relative to CWD).

    Returns:
        Validated :class:`Config` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If any field value is invalid.

    Example::

        cfg = load_config("configs/default.yaml")
        print(cfg.seed)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: '{path}'. "
            f"Ensure you are running from the repository root or provide an absolute path."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    return Config(**raw)
