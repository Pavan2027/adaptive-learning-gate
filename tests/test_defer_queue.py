"""
Tests for alg.gating.defer_queue (DeferQueue).

Covers:
- Basic enqueue and batch popping.
- Bounded capacity enforcement.
- Overflow policy 'drop_oldest'.
- Overflow policy 'reject_new'.
- Overflow policy 'drop_lowest_utility'.
- Queue clearing and statistics reporting.
"""

import pytest
import torch

from alg.data.stream import SampleMetadata, StreamSample
from alg.gating import DeferQueue, GateAction, GateDecision, ReasonCode
from alg.signals.types import SignalScores


def _create_dummy_item(sample_id: str, utility: float = 0.5) -> tuple[StreamSample, GateDecision]:
    sample = StreamSample(
        sample_id=sample_id,
        stream_index=0,
        image=torch.randn(3, 32, 32),
        observed_label=1,
        metadata=SampleMetadata(
            true_label=1,
            is_noisy_label=False,
            is_corrupted_feature=False,
            is_redundant=False,
            segment_id=0,
            original_dataset_index=0,
        ),
    )
    decision = GateDecision(
        sample_id=sample_id,
        stream_index=0,
        action=GateAction.DEFER,
        utility=utility,
        benefit=utility,
        harm=0.0,
        reason=ReasonCode.AMBIGUOUS_DEFERRED,
        reason_detail="Ambiguous utility",
        signals=SignalScores(0.5, 0.5, 0.5),
    )
    return sample, decision


class TestDeferQueue:
    def test_invalid_capacity_raises_error(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 0"):
            DeferQueue(capacity=-1)

    def test_invalid_policy_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown overflow_policy"):
            DeferQueue(capacity=10, overflow_policy="invalid_policy")

    def test_enqueue_and_pop_batch(self) -> None:
        queue = DeferQueue(capacity=10)
        s1, d1 = _create_dummy_item("s1")
        s2, d2 = _create_dummy_item("s2")

        assert queue.enqueue(s1, d1)
        assert queue.enqueue(s2, d2)
        assert len(queue) == 2

        popped = queue.pop_batch(batch_size=1)
        assert len(popped) == 1
        assert popped[0][0].sample_id == "s1"
        assert len(queue) == 1

        popped_rest = queue.pop_batch(batch_size=5)
        assert len(popped_rest) == 1
        assert len(queue) == 0

    def test_overflow_policy_drop_oldest(self) -> None:
        queue = DeferQueue(capacity=2, overflow_policy="drop_oldest")
        s1, d1 = _create_dummy_item("s1")
        s2, d2 = _create_dummy_item("s2")
        s3, d3 = _create_dummy_item("s3")

        queue.enqueue(s1, d1)
        queue.enqueue(s2, d2)
        assert queue.is_full

        # Enqueueing s3 should evict s1 (oldest)
        queue.enqueue(s3, d3)
        assert len(queue) == 2

        items = queue.pop_batch(batch_size=2)
        sample_ids = [item[0].sample_id for item in items]
        assert "s1" not in sample_ids
        assert sample_ids == ["s2", "s3"]
        assert queue.get_statistics()["overflow_drops"] == 1

    def test_overflow_policy_reject_new(self) -> None:
        queue = DeferQueue(capacity=2, overflow_policy="reject_new")
        s1, d1 = _create_dummy_item("s1")
        s2, d2 = _create_dummy_item("s2")
        s3, d3 = _create_dummy_item("s3")

        queue.enqueue(s1, d1)
        queue.enqueue(s2, d2)

        # Enqueueing s3 should be rejected
        accepted = queue.enqueue(s3, d3)
        assert not accepted
        assert len(queue) == 2
        items = queue.pop_batch(batch_size=2)
        assert [item[0].sample_id for item in items] == ["s1", "s2"]
        assert queue.get_statistics()["overflow_drops"] == 1

    def test_overflow_policy_drop_lowest_utility(self) -> None:
        queue = DeferQueue(capacity=2, overflow_policy="drop_lowest_utility")
        s_low, d_low = _create_dummy_item("s_low", utility=0.20)
        s_mid, d_mid = _create_dummy_item("s_mid", utility=0.50)
        s_high, d_high = _create_dummy_item("s_high", utility=0.55)

        queue.enqueue(s_low, d_low)
        queue.enqueue(s_mid, d_mid)

        # Enqueueing s_high should drop s_low (minimum utility 0.20)
        queue.enqueue(s_high, d_high)
        assert len(queue) == 2

        items = queue.pop_batch(batch_size=2)
        sample_ids = [item[0].sample_id for item in items]
        assert "s_low" not in sample_ids
        assert set(sample_ids) == {"s_mid", "s_high"}
