"""
Unit tests for Phase 8 plotting and analysis scripts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def dummy_results_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with dummy experiment runs for testing."""
    records = [
        {
            "run_id": "dummy_update_all",
            "method": "update_all",
            "method_name": "Conventional (Update-All)",
            "condition": "noisy_stream",
            "condition_name": "Standard Perturbed Stream",
            "seed": 42,
            "stream_size": 100,
            "final_test_accuracy": 70.0,
            "final_test_f1": 69.5,
            "total_updates": 5,
            "samples_learned": 100,
            "samples_ignored": 0,
            "samples_deferred": 0,
            "noise_ingested": 20,
            "total_noisy_labels": 20,
            "noise_interception_pct": 0.0,
            "sample_savings_pct": 0.0,
            "trajectory": [{"sample": 50, "test_acc": 0.70, "test_f1": 0.69, "update_steps": 2}],
        },
        {
            "run_id": "dummy_gated_full",
            "method": "gated_full",
            "method_name": "Adaptive Learning Gate (Ours)",
            "condition": "noisy_stream",
            "condition_name": "Standard Perturbed Stream",
            "seed": 42,
            "stream_size": 100,
            "final_test_accuracy": 72.0,
            "final_test_f1": 71.5,
            "total_updates": 3,
            "samples_learned": 50,
            "samples_ignored": 40,
            "samples_deferred": 10,
            "noise_ingested": 1,
            "total_noisy_labels": 20,
            "noise_interception_pct": 95.0,
            "sample_savings_pct": 50.0,
            "trajectory": [{"sample": 50, "test_acc": 0.72, "test_f1": 0.71, "update_steps": 2}],
        },
        {
            "run_id": "dummy_high_noise_update_all",
            "method": "update_all",
            "method_name": "Conventional (Update-All)",
            "condition": "high_noise_stream",
            "condition_name": "High Noise Stream",
            "seed": 42,
            "stream_size": 100,
            "final_test_accuracy": 55.0,
            "final_test_f1": 54.0,
            "total_updates": 5,
            "samples_learned": 100,
            "samples_ignored": 0,
            "samples_deferred": 0,
            "noise_ingested": 40,
            "total_noisy_labels": 40,
            "noise_interception_pct": 0.0,
            "sample_savings_pct": 0.0,
            "trajectory": [{"sample": 50, "test_acc": 0.58, "test_f1": 0.57, "update_steps": 2}],
        },
        {
            "run_id": "dummy_high_noise_gated_full",
            "method": "gated_full",
            "method_name": "Adaptive Learning Gate (Ours)",
            "condition": "high_noise_stream",
            "condition_name": "High Noise Stream",
            "seed": 42,
            "stream_size": 100,
            "final_test_accuracy": 66.0,
            "final_test_f1": 65.5,
            "total_updates": 3,
            "samples_learned": 45,
            "samples_ignored": 45,
            "samples_deferred": 10,
            "noise_ingested": 2,
            "total_noisy_labels": 40,
            "noise_interception_pct": 95.0,
            "sample_savings_pct": 55.0,
            "trajectory": [{"sample": 50, "test_acc": 0.67, "test_f1": 0.66, "update_steps": 2}],
        },
    ]

    for rec in records:
        out_f = tmp_path / f"{rec['run_id']}.json"
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(rec, f)

    return tmp_path


class TestPlottingPipeline:
    """Validate plot_results.py generation."""

    def test_plot_generation(self, dummy_results_dir: Path, tmp_path: Path) -> None:
        fig_dir = tmp_path / "figures"
        cmd = [
            sys.executable,
            "scripts/plot_results.py",
            "--input-dir",
            str(dummy_results_dir),
            "--output-dir",
            str(fig_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert result.returncode == 0
        assert (fig_dir / "fig1_stream_trajectory_high_noise.png").exists()
        assert (fig_dir / "fig2_pareto_noise_vs_compute.png").exists()
        assert (fig_dir / "fig4_noise_ingestion_comparison.png").exists()


class TestSummarizerPipeline:
    """Validate summarize_results.py generation."""

    def test_summary_generation(self, dummy_results_dir: Path, tmp_path: Path) -> None:
        summary_file = tmp_path / "test_summary.md"
        cmd = [
            sys.executable,
            "scripts/summarize_results.py",
            "--input-dir",
            str(dummy_results_dir),
            "--output-file",
            str(summary_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert result.returncode == 0
        assert summary_file.exists()
        content = summary_file.read_text(encoding="utf-8")
        assert "# Adaptive Learning Gate — Comprehensive Empirical Analysis Summary" in content
        assert "dummy_gated_full" in content
