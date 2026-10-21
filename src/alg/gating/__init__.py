"""
Adaptive Learning Gate modules and quarantine queue.

Exports:
- GateAction: Action enum (LEARN, IGNORE, DEFER).
- ReasonCode: Audit reason codes.
- GateDecision: Decision dataclass storing complete evaluation record.
- RuleBasedGatePolicy: Mathematical utility decision engine.
- UncertaintyGatePolicy: Active Learning baseline policy.
- RandomGatePolicy: Budget-matched random sampling baseline policy.
- DeferQueue: Bounded quarantine queue with overflow policies.
- AdaptiveLearningGate: Main gating coordinator.
"""

from alg.gating.decision import GateAction, GateDecision, ReasonCode
from alg.gating.defer_queue import DeferQueue
from alg.gating.gate import AdaptiveLearningGate
from alg.gating.policy import (
    RandomGatePolicy,
    RuleBasedGatePolicy,
    UncertaintyGatePolicy,
)

__all__ = [
    "GateAction",
    "ReasonCode",
    "GateDecision",
    "RuleBasedGatePolicy",
    "UncertaintyGatePolicy",
    "RandomGatePolicy",
    "DeferQueue",
    "AdaptiveLearningGate",
]
