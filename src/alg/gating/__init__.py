"""
Adaptive Learning Gate logic and defer queue.

Implemented in Phase 5 (Adaptive Gate).

Planned contents:
- GateAction enum: LEARN | IGNORE | DEFER
- Rule-based gate using utility score + configurable thresholds:
    utility(x) = estimated_benefit(x) - estimated_harm(x)
    if utility >= learn_threshold  → LEARN
    if utility <= ignore_threshold → IGNORE
    otherwise                      → DEFER
- Bounded defer queue with overflow handling
- Gate decision logging (every decision records action + reason)
"""
