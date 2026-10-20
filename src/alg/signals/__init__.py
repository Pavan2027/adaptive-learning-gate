"""
Per-sample learning signals for the Adaptive Learning Gate.

Exports:
- SignalScores: Data structure storing normalized [0.0, 1.0] signals.
- compute_predictive_uncertainty: Normalized Shannon entropy / margin.
- PrototypeMemory: Bounded class prototype memory for novelty estimation.
- compute_reliability: Composite reliability combining feature sanity and label consistency.
- SignalExtractor: Unified extractor computing all signals in a single forward pass.
"""

from alg.signals.extractor import SignalExtractor
from alg.signals.novelty import PrototypeMemory
from alg.signals.reliability import (
    compute_feature_integrity,
    compute_label_consistency,
    compute_reliability,
)
from alg.signals.types import SignalScores
from alg.signals.uncertainty import (
    compute_margin_uncertainty,
    compute_predictive_entropy,
    compute_predictive_uncertainty,
)

__all__ = [
    "SignalScores",
    "SignalExtractor",
    "PrototypeMemory",
    "compute_predictive_uncertainty",
    "compute_predictive_entropy",
    "compute_margin_uncertainty",
    "compute_feature_integrity",
    "compute_label_consistency",
    "compute_reliability",
]
