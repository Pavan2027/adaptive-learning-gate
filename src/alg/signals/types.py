"""
Data structures for per-sample learning signals.

All signals are normalized scalar scores strictly bounded within [0.0, 1.0].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignalScores:
    """
    Extracted learning signals for an incoming sample or batch.

    All scores are normalized to [0.0, 1.0]:
    - uncertainty : 0.0 = completely confident, 1.0 = maximum ambiguity / entropy
    - novelty     : 0.0 = identical to known prototypes, 1.0 = completely unfamiliar
    - reliability : 0.0 = highly suspicious / corrupt / mislabeled, 1.0 = clean and consistent
    """

    uncertainty: float
    novelty: float
    reliability: float
    raw_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert signal scores to dictionary."""
        return {
            "uncertainty": round(self.uncertainty, 4),
            "novelty": round(self.novelty, 4),
            "reliability": round(self.reliability, 4),
            "raw_metrics": {k: round(v, 4) for k, v in self.raw_metrics.items()},
        }
