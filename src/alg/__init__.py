"""
Adaptive Learning Gate (ALG)
============================

A research prototype for selective updates in continual neural network training.

The gate evaluates each incoming sample and assigns one of three actions:
  - LEARN  : use the sample for a model weight update
  - IGNORE : skip the sample (redundant, noisy, or low estimated value)
  - DEFER  : quarantine the sample in a bounded queue for later re-evaluation

Decision rule::

    utility(x) = estimated_benefit(x) - estimated_harm(x)

    if utility >= learn_threshold  → LEARN
    if utility <= ignore_threshold → IGNORE
    otherwise                     → DEFER

Subpackages
-----------
- config      : configuration loading and Pydantic validation
- data        : dataset loaders and stream wrappers (Phase 2+)
- models      : model architectures (Phase 2+)
- signals     : per-sample signal extractors (Phase 4+)
- gating      : adaptive learning gate and defer queue (Phase 5+)
- training    : training loops and update logic (Phase 2+)
- evaluation  : metrics and result logging (Phase 2+)
- utils       : shared utilities (seeding, logging helpers)
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
