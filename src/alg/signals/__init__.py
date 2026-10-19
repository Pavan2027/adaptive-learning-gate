"""
Per-sample signal extraction for the Adaptive Learning Gate.

Implemented in Phase 4 (Learning Signals).

Planned signal types:
- Predictive uncertainty  (e.g. softmax entropy, MC-dropout)
- Novelty                 (e.g. feature distance from running prototype)
- Reliability             (e.g. loss-based corruption indicator)
- Update-cost proxy       (optional; e.g. gradient norm estimate)

Each extractor returns a normalized score in [0.0, 1.0].
"""
