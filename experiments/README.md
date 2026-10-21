# Comparative Experiments & Ablations Suite

This directory documents the systematic experiment matrix and ablation studies for the research paper: **"Learning When Not to Learn: An Adaptive Gating Framework for Continual Neural Network Training"**.

---

## 1. Experiment Matrix Overview

All experiments evaluate models on a continuous stream of incoming samples. Baseline models start from the same pre-trained checkpoint ([`checkpoints/best_model.pt`](../checkpoints/best_model.pt)) and consume the exact same sequential stream manifest.

### Methods Compared:
1. **`update_all` (Conventional Continual Learning):**
   Standard continual learner updating unconditionally on 100% of stream samples.
2. **`random` (Budget-Matched Random Selection):**
   Selects samples uniformly at random with a budget probability $p = 0.35$ matching the compute budget of the full gate.
3. **`uncertainty_only` (Active Learning Baseline):**
   Selects samples based purely on predictive entropy ($U(x) \ge 0.35$). Demonstrates the classic vulnerability where label noise induces high uncertainty and tricks the model into updating on corrupted labels.
4. **`gated_full` (Adaptive Learning Gate - Proposed):**
   Full multi-signal gating evaluating net utility $\mathcal{U}(x) = (B(x) - H(x) + 1)/2$ with bounded defer queue quarantine and recheck.

### Component Ablations (One-Factor-at-a-Time):
1. **`ablation_no_novelty` ($w_{\text{nov}} = 0$):**
   Removes prototype cosine distance in latent feature space.
2. **`ablation_no_reliability` ($w_{\text{unrel}} = 0$, uncoupled):**
   Disables label-feature concordance check, testing behavior when noise is not actively penalized.
3. **`ablation_no_uncertainty` ($w_{\text{unc}} = 0$):**
   Removes predictive entropy from the benefit equation.
4. **`ablation_no_defer` (Hard Binary Gate):**
   Forces $\tau_{\text{learn}} = \tau_{\text{ignore}} = 0.40$, disallowing quarantine/deferral.

### Stream Conditions Evaluated:
- **`noisy_stream`:** 20% label noise, 15% feature corruption (Primary Stress Test).
- **`clean_stream`:** 0% noise, 0% corruption (Standard Continual Plasticity).
- **`redundant_stream`:** 30% sample repetition (Compute Efficiency Test).
- **`high_noise_stream`:** 40% label corruption (Extreme Adversity Stress Test).

---

## 2. Reproduction Commands

### Quick Sanity Run (Smoke Test):
```powershell
python scripts/run_experiments.py --smoke
```

### Run Primary Baselines on Noisy Stream (2,000 samples, Seed 42):
```powershell
python scripts/run_experiments.py --condition noisy_stream --methods update_all,random,uncertainty_only,gated_full --stream-size 2000
```

### Run Full Ablation Suite:
```powershell
python scripts/run_experiments.py --condition noisy_stream --run-ablations --stream-size 2000
```

### Multi-Seed Benchmark (Seeds 42, 43, 44):
```powershell
python scripts/run_experiments.py --condition noisy_stream --seeds 42,43,44 --stream-size 2000
```

### Aggregate Multi-Seed Results & Generate LaTeX Table:
```powershell
python scripts/aggregate_results.py --latex
```

---

## 3. Output Data Structure

- Individual run records: `results/experiments/{method}_{condition}_s{seed}.json`
- Multi-seed aggregated summaries: `research/data/aggregated_experiments.json`
- Schema fields:
  - `run_id`: Unique identifier
  - `final_test_accuracy`: Top-1 Accuracy on held-out test set (%)
  - `final_test_f1`: Macro-F1 score (%)
  - `noise_ingested`: Number of label-corrupted samples trained on
  - `noise_interception_pct`: Percentage of corrupted labels successfully blocked (%)
  - `sample_savings_pct`: Fraction of backward pass updates conserved (%)
  - `trajectory`: Step-by-step evaluation curve over stream progression
