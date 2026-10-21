"""
Evaluation metrics, statistical aggregation, and logging for the Adaptive Learning Gate.

Exports:
- compute_classification_metrics: Calculate accuracy, macro-F1, per-class metrics.
- evaluate_model: Full evaluation loop over a DataLoader.
- save_metrics_summary: Export dictionary to clean JSON.
- calculate_statistics: Compute mean, std, min, max across multi-seed runs.
- aggregate_experiment_runs: Aggregate JSON run results by condition and method.
"""

from alg.evaluation.aggregation import (
    aggregate_experiment_runs,
    calculate_statistics,
)
from alg.evaluation.metrics import (
    compute_classification_metrics,
    evaluate_model,
    save_metrics_summary,
)

__all__ = [
    "compute_classification_metrics",
    "evaluate_model",
    "save_metrics_summary",
    "calculate_statistics",
    "aggregate_experiment_runs",
]
