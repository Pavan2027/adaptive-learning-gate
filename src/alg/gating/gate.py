"""
AdaptiveLearningGate coordinator combining policy evaluation and quarantine queue.
"""

from __future__ import annotations

import logging
from typing import Any

from alg.data.stream import StreamSample
from alg.gating.decision import GateAction, GateDecision
from alg.gating.defer_queue import DeferQueue
from alg.gating.policy import RuleBasedGatePolicy
from alg.signals.types import SignalScores

logger = logging.getLogger(__name__)


class AdaptiveLearningGate:
    """
    Main gating module orchestrating sample evaluation, logging, and deferral.
    """

    def __init__(
        self,
        policy: RuleBasedGatePolicy | None = None,
        defer_queue: DeferQueue | None = None,
        learn_threshold: float = 0.50,
        ignore_threshold: float = 0.30,
        defer_queue_capacity: int = 500,
        overflow_policy: str = "drop_oldest",
    ) -> None:
        self.policy = policy or RuleBasedGatePolicy(
            learn_threshold=learn_threshold,
            ignore_threshold=ignore_threshold,
        )
        self.defer_queue = defer_queue or DeferQueue(
            capacity=defer_queue_capacity,
            overflow_policy=overflow_policy,
        )

        self.action_counts: dict[str, int] = {
            GateAction.LEARN.value: 0,
            GateAction.IGNORE.value: 0,
            GateAction.DEFER.value: 0,
        }
        self.total_decisions = 0

    def evaluate_sample(
        self,
        signals: SignalScores,
        sample: StreamSample | None = None,
        sample_id: str = "sample_000000",
        stream_index: int = 0,
    ) -> GateDecision:
        """
        Evaluate sample signals, assign action, and quarantine if deferred.

        Args:
            signals: Normalized SignalScores.
            sample: Optional StreamSample object (stored in queue if deferred).
            sample_id: String identifier.
            stream_index: Stream arrival position.

        Returns:
            Auditable GateDecision.
        """
        if sample is not None:
            sample_id = sample.sample_id
            stream_index = sample.stream_index

        decision = self.policy.evaluate(
            signals=signals,
            sample_id=sample_id,
            stream_index=stream_index,
        )

        self.action_counts[decision.action.value] += 1
        self.total_decisions += 1

        # Quarantine if deferred
        if decision.action == GateAction.DEFER and sample is not None:
            self.defer_queue.enqueue(sample, decision)

        return decision

    def get_summary_statistics(self) -> dict[str, Any]:
        """Return decision distribution and queue health metrics."""
        total = max(1, self.total_decisions)
        return {
            "total_decisions": self.total_decisions,
            "actions": {
                "learn": self.action_counts[GateAction.LEARN.value],
                "learn_percentage": round(self.action_counts[GateAction.LEARN.value] / total * 100, 2),
                "ignore": self.action_counts[GateAction.IGNORE.value],
                "ignore_percentage": round(self.action_counts[GateAction.IGNORE.value] / total * 100, 2),
                "defer": self.action_counts[GateAction.DEFER.value],
                "defer_percentage": round(self.action_counts[GateAction.DEFER.value] / total * 100, 2),
            },
            "queue": self.defer_queue.get_statistics(),
        }
