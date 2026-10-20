"""
Rule-Based Decision Policy for the Adaptive Learning Gate.

Mathematical Formulation:
1. Estimated Benefit:
   B(x) = w_nov * Novelty(x) + w_unc * (Uncertainty(x) * Reliability(x))
   - Novelty provides value (expanding model representation space).
   - Uncertainty is ONLY beneficial if Reliability is high (prevents learning from noise!).

2. Estimated Harm / Risk:
   H(x) = w_unrel * (1.0 - Reliability(x)) + w_red * ((1.0 - Novelty(x)) * (1.0 - Uncertainty(x)))
   - Low reliability creates high harm (risk of corrupting model weights).
   - High redundancy with low uncertainty creates waste (unnecessary compute).

3. Normalized Utility:
   Raw_Utility(x) = B(x) - H(x) in [-1.0, 1.0]
   Utility(x) = (Raw_Utility(x) + 1.0) / 2.0 in [0.0, 1.0]

Decision Rules:
- If Utility(x) >= learn_threshold  -> LEARN  (HIGH_UTILITY)
- If Utility(x) <= ignore_threshold -> IGNORE (LOW_RELIABILITY or REDUNDANT)
- Otherwise                         -> DEFER  (AMBIGUOUS_DEFERRED)
"""

from __future__ import annotations

import math
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
    ) -> None:
        if ignore_threshold >= learn_threshold:
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
        # Note: uncertainty only provides benefit when reliability is high
        effective_uncertainty = u * r
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
