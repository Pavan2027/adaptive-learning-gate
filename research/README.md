# Research Assets & Documentation

This directory holds materials and assets relevant to the research paper: **"Learning When Not to Learn: An Adaptive Gating Framework for Continual Neural Network Training"**.

## Contents & Organization

- `notes/`: Literature review summaries, mathematical formulations, and research hypotheses.
  - [`phase2_baseline_notes.md`](notes/phase2_baseline_notes.md): CIFAR-10 baseline run metrics, per-class accuracies, and training parameters.
  - [`phase3_stream_notes.md`](notes/phase3_stream_notes.md): Continual stream simulator mathematical formulations, perturbations, and fairness/zero-leakage design.
  - [`phase4_signals_notes.md`](notes/phase4_signals_notes.md): Learning signals mathematical formulations (Entropy uncertainty, Prototype novelty, and Reliability).
  - [`phase4_empirical_findings.md`](notes/phase4_empirical_findings.md): Empirical signal analysis proving the +0.7780 reliability separation gap and showing why uncertainty alone fails.
  - [`phase5_gate_notes.md`](notes/phase5_gate_notes.md): Adaptive Learning Gate utility formulation, decision policies, bounded defer queue, and 96.43% noise interception results.
  - [`phase6_continual_training_notes.md`](notes/phase6_continual_training_notes.md): Gated continual training pipeline architecture, update semantics, and conventional comparison.
- `plots/`: Generated figures, trade-off curves (e.g. accuracy vs. update fraction), and confusion matrices.
- `data/`: Exported summary metrics (JSON/CSV) from multi-seed runs across baselines and ablations.
  - [`baseline_cifar10_cnn.json`](data/baseline_cifar10_cnn.json): Canonical 5-epoch baseline evaluation metrics on CIFAR-10 (73.35% accuracy).
  - [`stream_manifest_summary.json`](data/stream_manifest_summary.json): Summary statistics of the 10,000-sample continual stream with 20% label noise and 15% feature corruption.
  - [`signal_analysis.json`](data/signal_analysis.json): Empirical signal distributions and separation metrics across 2,000 stream samples.
  - [`gate_evaluation.json`](data/gate_evaluation.json): Gate decision breakdown and efficacy metrics across 2,000 continuous stream samples.
- `draft/`: Working notes, outlines, and LaTeX snippets for the final manuscript.
