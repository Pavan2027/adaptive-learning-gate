"""
Evaluation metrics and logging for the Adaptive Learning Gate.

Exports:
- compute_classification_metrics: Calculate accuracy, macro-F1, per-class metrics.
- evaluate_model: Full evaluation loop over a DataLoader.
- save_metrics_summary: Export dictionary to clean JSON.
"""

from alg.evaluation.metrics import (
    compute_classification_metrics,
    evaluate_model,
    save_metrics_summary,
)

__all__ = ["compute_classification_metrics", "evaluate_model", "save_metrics_summary"]
