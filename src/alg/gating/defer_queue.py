"""
Bounded quarantine queue for deferred streaming samples.

Stores samples assigned the `DEFER` action along with their initial gate decision.
Provides configurable overflow policies to enforce bounded memory:
- 'drop_oldest'        : FIFO eviction when capacity is exceeded.
- 'drop_lowest_utility': Evicts the sample with the lowest utility score.
- 'reject_new'         : Rejects incoming samples when full.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from alg.data.stream import StreamSample
from alg.gating.decision import GateDecision

logger = logging.getLogger(__name__)


class DeferQueue:
    """
    Bounded FIFO/priority queue for quarantined training samples.
    """

    def __init__(
        self,
        capacity: int = 500,
        overflow_policy: str = "drop_oldest",
    ) -> None:
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}.")

        self.capacity = capacity
        self.overflow_policy = overflow_policy.lower()

        valid_policies = {"drop_oldest", "drop_lowest_utility", "reject_new"}
        if self.overflow_policy not in valid_policies:
            raise ValueError(
                f"Unknown overflow_policy '{overflow_policy}'. Supported: {list(valid_policies)}"
            )

        self._queue: list[tuple[StreamSample, GateDecision]] = []
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._overflow_drops = 0

    def __len__(self) -> int:
        return len(self._queue)

    @property
    def is_full(self) -> bool:
        """Check if queue is at capacity (if capacity > 0)."""
        return self.capacity > 0 and len(self._queue) >= self.capacity

    def enqueue(self, sample: StreamSample, decision: GateDecision) -> bool:
        """
        Attempt to add a deferred sample to the quarantine queue.

        Args:
            sample: The deferred StreamSample.
            decision: The GateDecision audit record.

        Returns:
            True if sample was enqueued, False if rejected due to overflow.
        """
        if self.is_full:
            if self.overflow_policy == "reject_new":
                self._overflow_drops += 1
                logger.debug("DeferQueue full: rejected sample '%s'.", sample.sample_id)
                return False

            elif self.overflow_policy == "drop_oldest":
                dropped = self._queue.pop(0)
                self._overflow_drops += 1
                logger.debug("DeferQueue full: dropped oldest sample '%s'.", dropped[0].sample_id)

            elif self.overflow_policy == "drop_lowest_utility":
                # Find index with minimum utility
                min_idx = min(range(len(self._queue)), key=lambda i: self._queue[i][1].utility)
                dropped = self._queue.pop(min_idx)
                self._overflow_drops += 1
                logger.debug(
                    "DeferQueue full: dropped lowest-utility sample '%s' (utility=%.3f).",
                    dropped[0].sample_id,
                    dropped[1].utility,
                )

        self._queue.append((sample, decision))
        self._total_enqueued += 1
        return True

    def pop_batch(self, batch_size: int = 32) -> list[tuple[StreamSample, GateDecision]]:
        """
        Dequeue up to `batch_size` samples for reconsideration.

        Args:
            batch_size: Maximum number of items to retrieve.

        Returns:
            List of (StreamSample, GateDecision) pairs.
        """
        count = min(batch_size, len(self._queue))
        popped = self._queue[:count]
        self._queue = self._queue[count:]
        self._total_dequeued += len(popped)
        return popped

    def clear(self) -> None:
        """Clear all items in the queue."""
        self._queue.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Return operational metrics for queue monitoring."""
        return {
            "current_size": len(self._queue),
            "capacity": self.capacity,
            "overflow_policy": self.overflow_policy,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "overflow_drops": self._overflow_drops,
        }
