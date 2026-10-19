"""
Evaluation metrics and model evaluation utilities for the Adaptive Learning Gate.

Calculates:
- Top-1 Accuracy
- Macro-averaged F1 Score
- Per-class accuracy
- Average cross-entropy loss
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10,
) -> dict[str, Any]:
    """
    Compute standard classification metrics from numpy arrays of targets and predictions.

    Args:
        y_true: 1D array of ground truth class integer labels.
        y_pred: 1D array of predicted class integer labels.
        num_classes: Total number of classes.

    Returns:
        Dictionary with keys:
            - 'accuracy': float in [0.0, 1.0]
            - 'macro_f1': float in [0.0, 1.0]
            - 'per_class_accuracy': list of float accuracies per class
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class_accuracy": [0.0] * num_classes}

    total_samples = len(y_true)
    correct_total = np.sum(y_true == y_pred)
    accuracy = float(correct_total / total_samples)

    f1_scores = []
    per_class_acc = []

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        # Precision & Recall
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

        class_total = np.sum(y_true == c)
        class_acc = float(tp / class_total) if class_total > 0 else 0.0
        per_class_acc.append(class_acc)

    macro_f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_accuracy": [round(a, 4) for a in per_class_acc],
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 10,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate a model over a DataLoader.

    Args:
        model: PyTorch model.
        dataloader: DataLoader yielding (images, targets).
        criterion: Loss function (e.g. nn.CrossEntropyLoss()).
        device: Evaluation device (CPU or CUDA).
        num_classes: Number of classes.
        max_batches: Optional batch limit for smoke testing.

    Returns:
        Dictionary containing 'loss', 'accuracy', 'macro_f1', and 'num_samples'.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = float(total_loss / total_samples) if total_samples > 0 else 0.0
    metrics = compute_classification_metrics(
        np.array(all_targets), np.array(all_preds), num_classes=num_classes
    )
    metrics["loss"] = round(avg_loss, 4)
    metrics["num_samples"] = total_samples
    return metrics


def save_metrics_summary(results: dict[str, Any], filepath: str | Path) -> None:
    """Save an experiment results summary dictionary to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
