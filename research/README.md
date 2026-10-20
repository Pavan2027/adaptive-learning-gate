# Research Assets & Documentation

This directory holds materials and assets relevant to the research paper: **"Learning When Not to Learn: An Adaptive Gating Framework for Continual Neural Network Training"**.

## Contents & Organization

- `notes/`: Literature review summaries, mathematical formulations, and research hypotheses.
  - [`phase2_baseline_notes.md`](notes/phase2_baseline_notes.md): CIFAR-10 baseline run metrics, per-class accuracies, and training parameters.
  - [`phase3_stream_notes.md`](notes/phase3_stream_notes.md): Continual stream simulator mathematical formulations, perturbations, and fairness/zero-leakage design.
  - [`phase4_signals_notes.md`](notes/phase4_signals_notes.md): Learning signals mathematical formulations (Entropy uncertainty, Prototype novelty, and Reliability).
- `plots/`: Generated figures, trade-off curves (e.g. accuracy vs. update fraction), and confusion matrices.
- `data/`: Exported summary metrics (JSON/CSV) from multi-seed runs across baselines and ablations.
  - [`baseline_cifar10_cnn.json`](data/baseline_cifar10_cnn.json): Canonical 5-epoch baseline evaluation metrics on CIFAR-10 (73.35% accuracy).
- `draft/`: Working notes, outlines, and LaTeX snippets for the final manuscript.
