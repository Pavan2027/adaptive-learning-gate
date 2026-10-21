"""
Unit tests for Phase 7 Experiment Suite and Result Aggregation.

Covers:
- Configuration validation for configs/experiments.yaml.
- UncertaintyGatePolicy behavior (Active Learning baseline).
- RandomGatePolicy reproducibility and budget behavior.
- Hard binary gate ablation (allow_no_defer=True).
- Result aggregation statistics (mean, std calculation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from alg.gating import (
    GateAction,
    RandomGatePolicy,
    ReasonCode,
    RuleBasedGatePolicy,
    UncertaintyGatePolicy,
)
from alg.evaluation import aggregate_experiment_runs, calculate_statistics
from alg.signals.types import SignalScores


class TestExperimentsConfigSchema:
    def test_yaml_config_structure(self) -> None:
        cfg_path = Path("configs/experiments.yaml")
        assert cfg_path.exists(), "configs/experiments.yaml must exist."

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "meta" in cfg
        assert "conditions" in cfg
        assert "methods" in cfg
        assert "ablations" in cfg

        # Assert primary methods present
        required_methods = {"update_all", "random", "uncertainty_only", "gated_full"}
        assert required_methods.issubset(set(cfg["methods"].keys()))

        # Assert key ablations present
        required_ablations = {
            "ablation_no_novelty",
            "ablation_no_reliability",
            "ablation_no_uncertainty",
            "ablation_no_defer",
        }
        assert required_ablations.issubset(set(cfg["ablations"].keys()))


class TestBaselinePolicies:
    def test_uncertainty_policy_thresholding(self) -> None:
        policy = UncertaintyGatePolicy(uncertainty_threshold=0.40)

        # High uncertainty -> LEARN
        s_high = SignalScores(uncertainty=0.75, novelty=0.1, reliability=0.1)
        dec_high = policy.evaluate(s_high)
        assert dec_high.action == GateAction.LEARN
        assert dec_high.reason == ReasonCode.HIGH_UNCERTAINTY

        # Low uncertainty -> IGNORE
        s_low = SignalScores(uncertainty=0.20, novelty=0.9, reliability=0.9)
        dec_low = policy.evaluate(s_low)
        assert dec_low.action == GateAction.IGNORE
        assert dec_low.reason == ReasonCode.LOW_UNCERTAINTY

    def test_random_policy_determinism(self) -> None:
        p1 = RandomGatePolicy(budget_rate=0.5, seed=123)
        p2 = RandomGatePolicy(budget_rate=0.5, seed=123)

        dummy_signals = SignalScores(uncertainty=0.5, novelty=0.5, reliability=0.5)

        decisions_1 = [p1.evaluate(dummy_signals).action for _ in range(10)]
        decisions_2 = [p2.evaluate(dummy_signals).action for _ in range(10)]

        assert decisions_1 == decisions_2

    def test_hard_gate_no_defer_ablation(self) -> None:
        # Hard binary gate: learn_threshold == ignore_threshold allowed if allow_no_defer=True
        policy = RuleBasedGatePolicy(
            learn_threshold=0.45,
            ignore_threshold=0.45,
            allow_no_defer=True,
        )

        # Marginal sample that normally defers
        s = SignalScores(uncertainty=0.5, novelty=0.4, reliability=0.6)
        dec = policy.evaluate(s)

        # Must be either LEARN or IGNORE, never DEFER
        assert dec.action in (GateAction.LEARN, GateAction.IGNORE)
        assert dec.action != GateAction.DEFER


class TestResultAggregation:
    def test_calc_stats_single_and_multi_value(self) -> None:
        # Single value
        s1 = calculate_statistics([72.5])
        assert s1["mean"] == 72.5
        assert s1["std"] == 0.0
        assert s1["count"] == 1

        # Multi-value
        s2 = calculate_statistics([70.0, 72.0, 74.0])
        assert s2["mean"] == 72.0
        assert round(s2["std"], 2) == 2.0
        assert s2["count"] == 3

    def test_aggregate_runs_from_directory(self, tmp_path: Path) -> None:
        run1 = {
            "run_id": "gated_full_noisy_s42",
            "method": "gated_full",
            "method_name": "Adaptive Learning Gate",
            "condition": "noisy_stream",
            "condition_name": "Standard Perturbed Stream",
            "seed": 42,
            "final_test_accuracy": 71.0,
            "final_test_f1": 70.5,
            "noise_ingested": 15,
            "noise_interception_pct": 96.0,
            "sample_savings_pct": 65.0,
            "elapsed_seconds": 12.0,
        }
        run2 = {
            "run_id": "gated_full_noisy_s43",
            "method": "gated_full",
            "method_name": "Adaptive Learning Gate",
            "condition": "noisy_stream",
            "condition_name": "Standard Perturbed Stream",
            "seed": 43,
            "final_test_accuracy": 72.0,
            "final_test_f1": 71.5,
            "noise_ingested": 17,
            "noise_interception_pct": 95.5,
            "sample_savings_pct": 66.0,
            "elapsed_seconds": 11.5,
        }

        (tmp_path / "run1.json").write_text(json.dumps(run1), encoding="utf-8")
        (tmp_path / "run2.json").write_text(json.dumps(run2), encoding="utf-8")

        agg = aggregate_experiment_runs(tmp_path)
        assert "noisy_stream" in agg
        assert "gated_full" in agg["noisy_stream"]

        m_stats = agg["noisy_stream"]["gated_full"]
        assert m_stats["num_seeds"] == 2
        assert m_stats["accuracy"]["mean"] == 71.5
