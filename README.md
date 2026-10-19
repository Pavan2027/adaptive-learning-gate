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

---

## 5. Development Status

- [x] **Phase 1: Foundation** — Project structure, Pydantic v2 configuration system, deterministic seeding utility, and test suite.
- [x] **Phase 2: Baseline** — Conventional training baseline on CIFAR-10 / Fashion-MNIST (update-on-every-sample) with best checkpointing and held-out test evaluation.
- [x] **Phase 3: Stream Simulator** — Continual streaming pipeline with controlled label noise, feature perturbations, redundancy, and segment shifts.
- [ ] **Phase 4: Learning Signals** — Per-sample uncertainty, novelty, and reliability extractors.
- [ ] **Phase 5: Adaptive Gate** — Rule-based utility gate and bounded deferral queue.
- [ ] **Phase 6: Continual Training** — Gated continual training loop with selective updates.
- [ ] **Phase 7: Experiments** — Multi-seed comparative evaluations against standard baselines.
- [ ] **Phase 8: Analysis** — Ablations, Pareto frontiers (accuracy vs. compute), visualizations, and paper reporting.