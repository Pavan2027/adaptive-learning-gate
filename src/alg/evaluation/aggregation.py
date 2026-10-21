"""
Result aggregation and multi-seed statistical calculation for ALG experiments.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def calculate_statistics(values: list[float]) -> dict[str, float]:
    """Calculate mean, standard deviation, min, max, and sample count."""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(arr),
    }


def aggregate_experiment_runs(results_dir: str | Path) -> dict[str, Any]:
    """
    Read all run JSON files from results_dir and compute statistics grouped by (condition, method).
    """
    p = Path(results_dir)
    if not p.exists():
        return {}

    run_files = list(p.glob("*.json"))
    if not run_files:
        return {}

    # Group runs by (condition, method)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for rf in run_files:
        with open(rf, "r", encoding="utf-8") as f:
            data = json.load(f)
        cond = data.get("condition", "unknown")
        method = data.get("method", "unknown")
        grouped[(cond, method)].append(data)

    aggregated: dict[str, Any] = {}

    for (cond, method), runs in grouped.items():
        if cond not in aggregated:
            aggregated[cond] = {}

        m_name = runs[0].get("method_name", method)
        cond_name = runs[0].get("condition_name", cond)

        accs = [r["final_test_accuracy"] for r in runs]
        f1s = [r["final_test_f1"] for r in runs]
        noise_ingested = [r["noise_ingested"] for r in runs]
        noise_intercept = [r["noise_interception_pct"] for r in runs]
        savings = [r["sample_savings_pct"] for r in runs]
        times = [r["elapsed_seconds"] for r in runs]

        aggregated[cond][method] = {
            "method_name": m_name,
            "condition_name": cond_name,
            "num_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "accuracy": calculate_statistics(accs),
            "macro_f1": calculate_statistics(f1s),
            "noise_ingested": calculate_statistics(noise_ingested),
            "noise_interception_pct": calculate_statistics(noise_intercept),
            "sample_savings_pct": calculate_statistics(savings),
            "elapsed_seconds": calculate_statistics(times),
        }

    return aggregated
