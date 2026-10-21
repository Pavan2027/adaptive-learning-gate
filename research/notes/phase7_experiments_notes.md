# Phase 7 — Comparative Experiments & Ablations: Matrix & Methodology

**Document ID:** `ALG-RES-007`  
**Phase:** 7 (Comparative Experiments and Ablations)  
**Status:** Framework Implemented & Smoke-Verified  
**Date:** 2026-10-21  
**Benchmark:** CIFAR-10 Continuous Stream Benchmark  

---

## 1. Executive Summary

Phase 7 implements the formal empirical evaluation framework for the paper: **"Learning When Not to Learn: An Adaptive Gating Framework for Continual Neural Network Training"**.

To establish empirical rigor, the evaluation enforces:
1. **Identical Stream Manifest:** All compared methods and ablations consume the exact same sequence of samples, perturbations, and random seeds.
2. **Identical Weight Initialization:** Every method starts from the exact same pre-trained checkpoint ([`checkpoints/best_model.pt`](../../checkpoints/best_model.pt)).
3. **Multi-Seed Statistical Aggregation:** Results report Mean $\pm$ Standard Deviation across random seeds.
4. **Machine-Readable Audit Trail:** Every run emits an individual JSON file (`results/experiments/{method}_{condition}_s{seed}.json`) containing full metrics and trajectories.

---

## 2. Compared Methods & Theoretical Baselines

| Method Identifier | Full Name | Policy / Gating Rule | Scientific Role / Hypotheses Tested |
|---|---|---|---|
| **`update_all`** | Conventional Continual Baseline | Unconditional updates on 100% of stream batches. | Standard continual learning baseline. Demonstrates vulnerability to label noise contamination. |
| **`random`** | Budget-Matched Random Selection | Selects samples with uniform probability $p = 0.35$. | Controls for compute budget. Proves that performance gains stem from informed signal selection, not simply updating on fewer samples. |
| **`uncertainty_only`** | Classical Active Learning | $\text{Action} = \text{LEARN}$ if $U(x) \ge 0.35$, else $\text{IGNORE}$. | Demonstrates the primary failure mode of active learning: label noise induces high predictive entropy, causing the model to actively select corrupted samples. |
| **`gated_full`** | Adaptive Learning Gate (Ours) | Multi-signal utility $\mathcal{U}(x) = (B(x) - H(x) + 1)/2$ with bounded deferral. | Proposed method combining uncertainty, novelty, reliability, and quarantine re-evaluation. |

---

## 3. Component Ablations (One-Factor-at-a-Time)

To isolate the contribution of each design component, the ablation suite toggles one component at a time relative to `gated_full`:

1. **`ablation_no_novelty` ($w_{\text{nov}} = 0.0$):**
   Disables prototype memory cosine distance. Tests how representation coverage suffers when the gate cannot identify novel class regions.
2. **`ablation_no_reliability` ($w_{\text{unrel}} = 0.0$, uncoupled):**
   Disables feature integrity and label consistency penalties. Tests the hypothesis that reliability is essential to prevent learning on corrupted data.
3. **`ablation_no_uncertainty` ($w_{\text{unc}} = 0.0$):**
   Removes Shannon entropy from benefit estimation. Tests sample efficiency when boundary uncertainty is ignored.
4. **`ablation_no_defer` (Hard Binary Gate):**
   Sets $\tau_{\text{learn}} = \tau_{\text{ignore}} = 0.40$ (`allow_no_defer=True`), forcing immediate binary decisions. Tests the value of quarantine and delayed re-evaluation.

---

## 4. Evaluated Stream Conditions

- **`noisy_stream`:** 20% label corruption, 15% feature noise (Primary benchmark stress test).
- **`clean_stream`:** 0% noise, 0% corruption (Tests plasticity and retention on uncorrupted streams).
- **`redundant_stream`:** 30% sample repetition (Tests compute conservation when familiar samples dominate).
- **`high_noise_stream`:** 40% label corruption (Stress tests gate robustness under extreme noise).

---

## 5. Automated Execution & Aggregation Pipeline

- **Configuration:** [`configs/experiments.yaml`](../../configs/experiments.yaml)
- **Experiment Runner:** [`scripts/run_experiments.py`](../../scripts/run_experiments.py)
- **Results Aggregator:** [`scripts/aggregate_results.py`](../../scripts/aggregate_results.py)
- **Reusable Aggregation Library:** [`src/alg/evaluation/aggregation.py`](../../src/alg/evaluation/aggregation.py)
- **Test Suite:** [`tests/test_experiment_results.py`](../../tests/test_experiment_results.py) (All 117 tests passing).
