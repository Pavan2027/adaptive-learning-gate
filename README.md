# Learning When Not to Learn: An Adaptive Gating Framework for Continual Neural Network Training

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An experimental research prototype for selective updates in continual neural network training.

---

## 1. Research Overview

Instead of updating the model on every incoming sample in a data stream, the **Adaptive Learning Gate (ALG)** evaluates each sample along multiple dimensions (uncertainty, novelty, and reliability) and assigns one of three discrete actions:

- **LEARN:** Update model parameters using the sample.
- **IGNORE:** Skip the sample (redundant, highly noisy, or low estimated utility).
- **DEFER:** Quarantine the sample in a bounded queue for reconsideration or batch re-evaluation.

### Decision Rule
A scalar utility score is computed from estimated signals:
$$\text{utility}(x) = \text{estimated\_benefit}(x) - \text{estimated\_harm}(x)$$

- $\text{utility}(x) \ge \text{learn\_threshold} \implies \textbf{LEARN}$
- $\text{utility}(x) \le \text{ignore\_threshold} \implies \textbf{IGNORE}$
- $\text{ignore\_threshold} < \text{utility}(x) < \text{learn\_threshold} \implies \textbf{DEFER}$

---

## 2. Repository Structure

```text
adaptive-learning-gate/
├── configs/
│   └── default.yaml         # Master YAML configuration file
├── src/
│   └── alg/                 # Core Python package
│       ├── __init__.py
│       ├── config.py        # Pydantic v2 configuration schema & loader
│       ├── data/            # Dataset loaders & stream generators (Phase 2+)
│       ├── models/          # Neural network architectures (Phase 2+)
│       ├── signals/         # Uncertainty, novelty & reliability extractors (Phase 4+)
│       ├── gating/          # Decision gate & deferral queue logic (Phase 5+)
│       ├── training/        # Continual training loops & baselines (Phase 2, 6+)
│       ├── evaluation/      # Metrics, retention & forgetting tracking (Phase 2+)
│       └── utils/
│           ├── __init__.py
│           └── seeding.py   # Deterministic seeding for Python, NumPy, PyTorch (CPU/CUDA)
├── scripts/
│   └── show_config.py       # Configuration inspection CLI
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   ├── test_config.py       # Configuration loading & validation tests
│   └── test_seeding.py      # Seed utility reproducibility tests
├── research/                # Paper assets, notes, plots, and analysis JSONs
├── agents/                  # Design specification documents & phase roadmap
├── pyproject.toml           # Package metadata and build system
├── requirements.txt         # Core dependencies
└── requirements-dev.txt     # Test and development dependencies
```

---

## 3. Environment Setup

### Step 1: Create and Activate Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install PyTorch with GPU Support
Install the PyTorch build that matches your local CUDA installation **before** other dependencies:

```powershell
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU-only fallback:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify your GPU is recognized:
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Step 3: Install Package & Dependencies
Install in editable mode along with development/testing requirements:
```powershell
pip install -e ".[dev]"
```
Or via requirements files:
```powershell
pip install -r requirements-dev.txt
```

---

## 4. Running Tests & Utilities

### Run the Test Suite
```powershell
pytest
```
Or with verbose output and coverage:
```powershell
pytest -v
```

### Inspect Configuration
Display the parsed and validated default configuration:
```powershell
python scripts/show_config.py
```
Display as JSON:
```powershell
python scripts/show_config.py --json
```
Validate and inspect a custom config:
```powershell
python scripts/show_config.py --config path/to/custom.yaml
```

### Train Conventional Baseline
Train the reference model (update-on-every-sample) on CIFAR-10 with GPU:
```powershell
python scripts/train_baseline.py
```
Or run a fast CPU smoke-test:
```powershell
python scripts/train_baseline.py --smoke-test --device cpu
```

### Inspect Continual Data Stream (Phase 3)
Generate and inspect a simulated stream with synthetic label noise and feature corruption:
```powershell
python scripts/inspect_stream.py --label-noise 0.20 --feature-corruption 0.15 --export-manifest
```

### Analyze Learning Signals (Phase 4)
Evaluate predictive uncertainty, prototype novelty, and reliability on the stream:
```powershell
python scripts/analyze_signals.py --stream-size 2000
```

### Evaluate Adaptive Learning Gate (Phase 5)
Run the decision policy and bounded quarantine queue on streaming inputs:
```powershell
python scripts/evaluate_gate.py --stream-size 2000
```

### Continual Streaming Adaptation (Phase 6)
Run selective continual learning with the adaptive gate:
```powershell
python scripts/run_stream.py --mode compare --stream-size 2000 --batch-size 64
```

### Run Systematic Experiment Matrix & Ablations (Phase 7)
Execute the complete comparative benchmark and component ablation suite:
```powershell
# Run primary comparison methods on perturbed stream
python scripts/run_experiments.py --condition noisy_stream --methods update_all,random,uncertainty_only,gated_full --stream-size 2000

# Run full component ablations
python scripts/run_experiments.py --condition noisy_stream --run-ablations --stream-size 2000

# Run extreme noise stress test (40% noise)
python scripts/run_experiments.py --condition high_noise_stream --methods update_all,uncertainty_only,gated_full --stream-size 2000

# Aggregate statistical LaTeX and Markdown tables
python scripts/aggregate_results.py --latex
```

### Generate Visualizations & Research Summary (Phase 8)
Generate 300 DPI publication-ready figures and comprehensive markdown reports:
```powershell
# Generate all publication figures into results/figures and research/plots
python scripts/plot_results.py

# Generate statistical summary report
python scripts/summarize_results.py
```

---

## 5. Development Status

- [x] **Phase 1: Foundation** — Project structure, Pydantic v2 configuration system, deterministic seeding utility, and test suite.
- [x] **Phase 2: Baseline** — Conventional training baseline on CIFAR-10 (update-on-every-sample) with best checkpointing and held-out test evaluation.
- [x] **Phase 3: Stream Simulator** — Continual streaming pipeline with controlled label noise, feature perturbations, redundancy, and segment shifts.
- [x] **Phase 4: Learning Signals** — Per-sample predictive uncertainty, prototype novelty, and composite reliability extractors.
- [x] **Phase 5: Adaptive Gate** — Multi-signal utility gate, auditable decision policies, and bounded quarantine deferral queue.
- [x] **Phase 6: Continual Training** — Gated continual training loop with selective updates and memory centroid updates.
- [x] **Phase 7: Experiments** — Systematic comparative benchmarks across stream conditions and one-factor-at-a-time component ablations.
- [x] **Phase 8: Analysis** — 300 DPI publication figures, Pareto frontiers, statistical summaries, and full research paper draft.