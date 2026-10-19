"""
Tests for alg.evaluation.metrics.

Covers:
- Metric calculation accuracy (perfect match, partial match).
- Macro-F1 computation across balanced and imbalanced predictions.
- Empty array edge case.
- Per-class accuracy computation.
"""

import numpy as np
import pytest

from alg.evaluation.metrics import compute_classification_metrics


class TestComputeClassificationMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=5)

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert all(acc == 1.0 for acc in metrics["per_class_accuracy"])

    def test_zero_accuracy(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=2)

        assert metrics["accuracy"] == 0.0
        assert metrics["macro_f1"] == 0.0

    def test_partial_accuracy(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=2)

        assert metrics["accuracy"] == 0.75
        assert 0.0 < metrics["macro_f1"] < 1.0

    def test_empty_arrays_handled_safely(self) -> None:
        y_true = np.array([])
        y_pred = np.array([])
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=3)

        assert metrics["accuracy"] == 0.0
        assert metrics["macro_f1"] == 0.0
        assert len(metrics["per_class_accuracy"]) == 3

    def test_per_class_accuracy(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1])
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=2)

        # Class 0: 2/2 = 1.0; Class 1: 1/2 = 0.5
        assert metrics["per_class_accuracy"] == [1.0, 0.5]
