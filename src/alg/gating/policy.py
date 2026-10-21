"""
Decision Policies for the Adaptive Learning Gate and Comparative Baselines.

Includes:
1. `RuleBasedGatePolicy`:
   The multi-signal decision engine evaluating utility U(x) from benefit B(x) and harm H(x).
   Supports ablation flags:
   - `w_novelty=0.0` (ablation: no novelty)
   - `w_uncertainty=0.0` (ablation: no uncertainty)
   - `w_unreliability=0.0`, `use_reliability_coupling=False` (ablation: no reliability)
   - `allow_no_defer=True`, `learn_threshold=0.40, ignore_threshold=0.40` (ablation: hard binary gate, no deferral)

2. `UncertaintyGatePolicy`:
   The classical Active Learning baseline: selects samples purely when predictive
   entropy exceeds a configured threshold, demonstrating susceptibility to label noise.

3. `RandomGatePolicy`:
   The budget-matched random baseline: admits samples with a uniform probability
   matching the compute budget of the full gate.
"""

from __future__ import annotations

import math
import random
from typing import Any

from alg.gating.decision import GateAction, GateDecision, ReasonCode
from alg.signals.types import SignalScores


class RuleBasedGatePolicy:
    """
    Transparent, auditable rule-based gating policy evaluating sample utility.
    """

    def __init__(
        self,
        learn_threshold: float = 0.50,
        ignore_threshold: float = 0.30,
        w_novelty: float = 0.5,
        w_uncertainty: float = 0.5,
        w_unreliability: float = 0.8,
        w_redundancy: float = 0.2,
        allow_no_defer: bool = False,
        use_reliability_coupling: bool = True,
    ) -> None:
        if not allow_no_defer and ignore_threshold >= learn_threshold:
            raise ValueError(
                f"ignore_threshold ({ignore_threshold}) must be strictly less than "
                f"learn_threshold ({learn_threshold}) to maintain a valid DEFER interval."
            )

        self.learn_threshold = learn_threshold
        self.ignore_threshold = ignore_threshold
        self.w_novelty = w_novelty
        self.w_uncertainty = w_uncertainty
        self.w_unreliability = w_unreliability
        self.w_redundancy = w_redundancy
        self.allow_no_defer = allow_no_defer
        self.use_reliability_coupling = use_reliability_coupling

    def evaluate(
        self,
        signals: SignalScores,
        sample_id: str = "sample_000000",
        stream_index: int = 0,
    ) -> GateDecision:
        """
        Evaluate learning signals and produce an auditable GateDecision.

        Args:
            signals: Normalized SignalScores for the sample.
            sample_id: Unique string identifier for the sample.
            stream_index: Arrival position in the continuous stream.

        Returns:
            GateDecision containing action, utility score, benefit, harm, and reason.
        """
        u = signals.uncertainty
        n = signals.novelty
        r = signals.reliability

        # 1. Fail-Safe Guard against non-finite values
        if not (math.isfinite(u) and math.isfinite(n) and math.isfinite(r)):
            return GateDecision(
                sample_id=sample_id,
                stream_index=stream_index,
                action=GateAction.IGNORE,
                utility=0.0,
                benefit=0.0,
                harm=1.0,
                reason=ReasonCode.FALLBACK_SAFE,
                reason_detail="Non-finite or invalid signal encountered; defaulted to IGNORE.",
                signals=signals,
            )

        # 2. Compute Benefit B(x)
        # Note: In full gate, uncertainty only provides benefit when reliability is high
        effective_uncertainty = (u * r) if self.use_reliability_coupling else u
        benefit = self.w_novelty * n + self.w_uncertainty * effective_uncertainty

        # 3. Compute Harm H(x)
        unreliability_harm = (1.0 - r) * self.w_unreliability
        # Redundancy penalty: sample is familiar (low novelty) AND already well-predicted (low uncertainty)
        redundancy_penalty = (1.0 - n) * (1.0 - u) * self.w_redundancy
        harm = unreliability_harm + redundancy_penalty

        # 4. Compute Normalized Utility in [0.0, 1.0]
        raw_utility = benefit - harm
        # Rescale [-1.0, 1.0] to [0.0, 1.0]
        utility = (raw_utility + 1.0) / 2.0
        utility = max(0.0, min(1.0, utility))

        # 5. Determine Action and Reason Code
        if utility >= self.learn_threshold:
            action = GateAction.LEARN
            reason = ReasonCode.HIGH_UTILITY
            detail = f"Utility {utility:.3f} >= learn_threshold {self.learn_threshold:.3f} (Benefit: {benefit:.3f}, Harm: {harm:.3f})."

        elif utility <= self.ignore_threshold:
            action = GateAction.IGNORE
            if r < 0.35:
                reason = ReasonCode.LOW_RELIABILITY
                detail = f"Low reliability {r:.3f} (suspected noise/corruption); Utility {utility:.3f} <= {self.ignore_threshold:.3f}."
            else:
                reason = ReasonCode.REDUNDANT
                detail = f"Redundant/uninformative sample; Utility {utility:.3f} <= {self.ignore_threshold:.3f}."

        else:
            action = GateAction.DEFER
            reason = ReasonCode.AMBIGUOUS_DEFERRED
            detail = f"Ambiguous utility {utility:.3f} in quarantine zone ({self.ignore_threshold:.3f}, {self.learn_threshold:.3f})."

        return GateDecision(
            sample_id=sample_id,
            stream_index=stream_index,
            action=action,
            utility=utility,
            benefit=benefit,
            harm=harm,
            reason=reason,
            reason_detail=detail,
            signals=signals,
        )


class UncertaintyGatePolicy:
    """
    Classical Active Learning baseline policy.
    Selects samples for training purely based on predictive entropy uncertainty.
    """

    def __init__(self, uncertainty_threshold: float = 0.35) -> None:
        self.uncertainty_threshold = uncertainty_threshold

    def evaluate(
        self,
        signals: SignalScores,
        sample_id: str = "sample_000000",
        stream_index: int = 0,
    ) -> GateDecision:
        u = signals.uncertainty
        if not math.isfinite(u):
            return GateDecision(
                sample_id=sample_id,
                stream_index=stream_index,
                action=GateAction.IGNORE,
                utility=0.0,
                benefit=0.0,
                harm=1.0,
                reason=ReasonCode.FALLBACK_SAFE,
                reason_detail="Non-finite uncertainty signal encountered.",
                signals=signals,
            )

        if u >= self.uncertainty_threshold:
            action = GateAction.LEARN
            reason = ReasonCode.HIGH_UNCERTAINTY
            detail = f"Uncertainty {u:.3f} >= threshold {self.uncertainty_threshold:.3f}."
        else:
            action = GateAction.IGNORE
            reason = ReasonCode.LOW_UNCERTAINTY
            detail = f"Uncertainty {u:.3f} < threshold {self.uncertainty_threshold:.3f}."

        return GateDecision(
            sample_id=sample_id,
            stream_index=stream_index,
            action=action,
            utility=u,
            benefit=u,
            harm=0.0,
            reason=reason,
            reason_detail=detail,
            signals=signals,
        )


class RandomGatePolicy:
    """
    Budget-matched random sampling baseline policy.
    Selects samples uniformly at random with probability equal to budget_rate.
    """

    def __init__(self, budget_rate: float = 0.35, seed: int | None = None) -> None:
        self.budget_rate = budget_rate
        self.rng = random.Random(seed)

    def evaluate(
        self,
        signals: SignalScores,
        sample_id: str = "sample_000000",
        stream_index: int = 0,
    ) -> GateDecision:
        roll = self.rng.random()
        if roll < self.budget_rate:
            action = GateAction.LEARN
            reason = ReasonCode.RANDOM_SELECTED
            detail = f"Random roll {roll:.3f} < budget {self.budget_rate:.3f}."
        else:
            action = GateAction.IGNORE
            reason = ReasonCode.RANDOM_SKIPPED
            detail = f"Random roll {roll:.3f} >= budget {self.budget_rate:.3f}."

        return GateDecision(
            sample_id=sample_id,
            stream_index=stream_index,
            action=action,
            utility=self.budget_rate,
            benefit=self.budget_rate,
            harm=0.0,
            reason=reason,
            reason_detail=detail,
            signals=signals,
        )
