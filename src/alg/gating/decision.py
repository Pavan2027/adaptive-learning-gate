"""
Typed action definitions, reason codes, and decision results for the Adaptive Learning Gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from alg.signals.types import SignalScores


class GateAction(str, Enum):
    """The three discrete actions the Adaptive Learning Gate can assign."""

    LEARN = "LEARN"    # Use sample for model parameter update
    IGNORE = "IGNORE"  # Skip sample (redundant, highly noisy, or harmful)
    DEFER = "DEFER"    # Quarantine sample in bounded queue for later reconsideration


class ReasonCode(str, Enum):
    """Auditable justification for the gate's decision."""

    HIGH_UTILITY = "HIGH_UTILITY"              # High informative value and high reliability
    LOW_RELIABILITY = "LOW_RELIABILITY"        # Suspected label noise or severe feature corruption
    REDUNDANT = "REDUNDANT"                    # Low novelty and low uncertainty (redundant compute)
    AMBIGUOUS_DEFERRED = "AMBIGUOUS_DEFERRED"  # Intermediate utility; requires quarantine/re-evaluation
    FALLBACK_SAFE = "FALLBACK_SAFE"            # Non-finite or missing signals safely ignored


@dataclass
class GateDecision:
    """
    Complete audit record for a single sample evaluated by the gate.
    """

    sample_id: str
    stream_index: int
    action: GateAction
    utility: float
    benefit: float
    harm: float
    reason: ReasonCode
    reason_detail: str
    signals: SignalScores

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to serializable dictionary."""
        return {
            "sample_id": self.sample_id,
            "stream_index": self.stream_index,
            "action": self.action.value,
            "utility": round(self.utility, 4),
            "benefit": round(self.benefit, 4),
            "harm": round(self.harm, 4),
            "reason": self.reason.value,
            "reason_detail": self.reason_detail,
            "signals": self.signals.to_dict(),
        }
