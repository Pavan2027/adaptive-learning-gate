"""
Tests for alg.gating (RuleBasedGatePolicy, AdaptiveLearningGate, and Decision objects).

Covers:
- Threshold validation (ignore < learn).
- High-utility sample -> LEARN.
- Low-reliability sample -> IGNORE.
- Critical requirement: High uncertainty + low reliability must NOT learn.
- Redundant sample -> IGNORE (compute saving).
- Intermediate sample -> DEFER.
- Non-finite fallback safety.
- Coordinator action tracking.
"""

import pytest

from alg.gating import (
    AdaptiveLearningGate,
    GateAction,
    ReasonCode,
    RuleBasedGatePolicy,
)
from alg.signals.types import SignalScores


class TestRuleBasedGatePolicy:
    def test_invalid_thresholds_raise_error(self) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            RuleBasedGatePolicy(learn_threshold=0.4, ignore_threshold=0.6)

    def test_equal_thresholds_raise_error(self) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            RuleBasedGatePolicy(learn_threshold=0.5, ignore_threshold=0.5)

    def test_high_utility_leads_to_learn(self) -> None:
        policy = RuleBasedGatePolicy(learn_threshold=0.60, ignore_threshold=0.35)
        # High novelty, high reliability, moderate uncertainty
        signals = SignalScores(uncertainty=0.5, novelty=0.8, reliability=0.9)
        decision = policy.evaluate(signals)

        assert decision.action == GateAction.LEARN
        assert decision.reason == ReasonCode.HIGH_UTILITY
        assert decision.utility >= 0.60

    def test_low_reliability_leads_to_ignore(self) -> None:
        policy = RuleBasedGatePolicy(learn_threshold=0.60, ignore_threshold=0.35)
        # Near zero reliability (e.g. noisy label or corrupt feature)
        signals = SignalScores(uncertainty=0.4, novelty=0.3, reliability=0.05)
        decision = policy.evaluate(signals)

        assert decision.action == GateAction.IGNORE
        assert decision.reason == ReasonCode.LOW_RELIABILITY
        assert decision.utility <= 0.35

    def test_critical_high_uncertainty_low_reliability_never_learns(self) -> None:
        """
        CRITICAL TEST:
        Under traditional active learning, high uncertainty alone would trigger training.
        Under ALG, a sample with high uncertainty but low reliability MUST NOT be learned.
        """
        policy = RuleBasedGatePolicy(learn_threshold=0.60, ignore_threshold=0.35)
        signals = SignalScores(uncertainty=0.95, novelty=0.5, reliability=0.05)
        decision = policy.evaluate(signals)

        # Must be IGNORE or DEFER, NEVER LEARN
        assert decision.action != GateAction.LEARN
        assert decision.action == GateAction.IGNORE

    def test_redundant_familiar_sample_ignored(self) -> None:
        policy = RuleBasedGatePolicy(learn_threshold=0.60, ignore_threshold=0.35)
        # Low novelty (identical to prototype) and zero uncertainty (already confident)
        signals = SignalScores(uncertainty=0.05, novelty=0.05, reliability=0.95)
        decision = policy.evaluate(signals)

        # Benefit is low, redundant penalty kicks in -> IGNORE or DEFER
        assert decision.action in (GateAction.IGNORE, GateAction.DEFER)
        assert decision.action != GateAction.LEARN

    def test_ambiguous_sample_deferred(self) -> None:
        policy = RuleBasedGatePolicy(learn_threshold=0.65, ignore_threshold=0.30)
        # Mid-level uncertainty, mid-level novelty, mid-level reliability
        signals = SignalScores(uncertainty=0.5, novelty=0.4, reliability=0.6)
        decision = policy.evaluate(signals)

        assert decision.action == GateAction.DEFER
        assert decision.reason == ReasonCode.AMBIGUOUS_DEFERRED
        assert 0.30 < decision.utility < 0.65

    def test_non_finite_signal_fallback(self) -> None:
        policy = RuleBasedGatePolicy()
        signals = SignalScores(uncertainty=float("nan"), novelty=0.5, reliability=0.5)
        decision = policy.evaluate(signals)

        assert decision.action == GateAction.IGNORE
        assert decision.reason == ReasonCode.FALLBACK_SAFE
        assert decision.utility == 0.0


class TestAdaptiveLearningGate:
    def test_action_counts_accumulate(self) -> None:
        gate = AdaptiveLearningGate(learn_threshold=0.60, ignore_threshold=0.35)

        s_learn = SignalScores(uncertainty=0.5, novelty=0.8, reliability=0.9)
        s_ignore = SignalScores(uncertainty=0.2, novelty=0.1, reliability=0.1)

        d1 = gate.evaluate_sample(s_learn)
        d2 = gate.evaluate_sample(s_ignore)

        assert d1.action == GateAction.LEARN
        assert d2.action == GateAction.IGNORE

        stats = gate.get_summary_statistics()
        assert stats["total_decisions"] == 2
        assert stats["actions"]["learn"] == 1
        assert stats["actions"]["ignore"] == 1
