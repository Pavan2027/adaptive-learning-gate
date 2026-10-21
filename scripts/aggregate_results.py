#!/usr/bin/env python3
"""
Aggregate experiment JSON results across multiple seeds and export summary tables.

Reads from `results/experiments/*.json`, groups runs by condition and method,
computes Mean +/- Std for all metrics, and exports:
1. Terminal Markdown Table
2. Publication LaTeX Table snippet
3. Machine-readable JSON summary to `research/data/aggregated_experiments.json`

Usage::

    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results-dir results/experiments --export research/data/aggregated_experiments.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alg.evaluation import aggregate_experiment_runs

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("aggregator")


def print_markdown_summary(aggregated: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("   Aggregated Benchmark Results across Seeds (Mean +/- Std)")
    print("=" * 88)

    for cond, methods in aggregated.items():
        cond_title = next(iter(methods.values()))["condition_name"]
        print(f"\nCondition: {cond_title} ({cond})")
        print(f"| {'Method':<32} | {'Seeds':>5} | {'Test Accuracy (%)':>19} | {'Noise Intercept (%)':>20} | {'Compute Saved (%)':>18} |")
        print(f"|{'-' * 34}|{'-' * 7}|{'-' * 21}|{'-' * 22}|{'-' * 20}|")

        for m_key, m_stats in methods.items():
            name = m_stats["method_name"]
            n_seeds = m_stats["num_seeds"]
            acc_m = m_stats["accuracy"]["mean"]
            acc_s = m_stats["accuracy"]["std"]
            acc_str = f"{acc_m:.2f} +/- {acc_s:.2f}" if n_seeds > 1 else f"{acc_m:.2f}"

            ni_m = m_stats["noise_interception_pct"]["mean"]
            ni_s = m_stats["noise_interception_pct"]["std"]
            ni_str = f"{ni_m:.1f} +/- {ni_s:.1f}" if n_seeds > 1 else f"{ni_m:.1f}"

            sav_m = m_stats["sample_savings_pct"]["mean"]
            sav_s = m_stats["sample_savings_pct"]["std"]
            sav_str = f"{sav_m:.1f} +/- {sav_s:.1f}" if n_seeds > 1 else f"{sav_m:.1f}"

            print(f"| {name:<32} | {n_seeds:>5} | {acc_str:>19} | {ni_str:>20} | {sav_str:>18} |")

    print("=" * 88 + "\n")


def print_latex_snippet(aggregated: dict[str, Any]) -> None:
    print("% -- LaTeX Table Snippet --------------------------------------------------------")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Comparative Continual Learning Performance on Perturbed Stream}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Seeds} & \textbf{Test Acc (\%)} & \textbf{Noise Blocked (\%)} & \textbf{Compute Saved (\%)} \\")
    print(r"\midrule")

    for cond, methods in aggregated.items():
        for m_key, m_stats in methods.items():
            name = m_stats["method_name"].replace("_", r"\_").replace("&", r"\&")
            n_seeds = m_stats["num_seeds"]
            acc_m = m_stats["accuracy"]["mean"]
            acc_s = m_stats["accuracy"]["std"]
            acc_str = f"${acc_m:.2f} \\pm {acc_s:.2f}$" if n_seeds > 1 else f"${acc_m:.2f}$"

            ni_m = m_stats["noise_interception_pct"]["mean"]
            ni_s = m_stats["noise_interception_pct"]["std"]
            ni_str = f"${ni_m:.1f} \\pm {ni_s:.1f}$" if n_seeds > 1 else f"${ni_m:.1f}$"

            sav_m = m_stats["sample_savings_pct"]["mean"]
            sav_str = f"${sav_m:.1f}$"

            print(f"{name} & {n_seeds} & {acc_str} & {ni_str} & {sav_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:continual_comparison}")
    print(r"\end{table}")
    print("% ------------------------------------------------------------------------------")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment runs.")
    parser.add_argument("--results-dir", default="results/experiments", help="Path to experiment JSONs.")
    parser.add_argument("--export", default="research/data/aggregated_experiments.json", help="Path to export JSON.")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table.")
    args = parser.parse_args()

    agg = aggregate_experiment_runs(args.results_dir)
    if not agg:
        print("No results to aggregate.")
        return

    print_markdown_summary(agg)

    if args.latex:
        print_latex_snippet(agg)

    out_file = Path(args.export)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print(f"Aggregated results saved to: {out_file.resolve()}\n")


if __name__ == "__main__":
    main()
