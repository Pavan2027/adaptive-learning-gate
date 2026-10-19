"""
Seed utility for reproducible experiments.

Sets random seeds for Python's ``random`` module, NumPy, and PyTorch
(including CUDA if a GPU is available). If PyTorch is not installed,
a warning is logged and execution continues — all Phase 1 tests pass
without PyTorch being present.

Usage::

    from alg.utils.seeding import set_seed

    set_seed(42)
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for fully deterministic execution.

    Sets seeds for:
    - Python's built-in ``random`` module
    - NumPy
    - PyTorch CPU (if installed)
    - PyTorch CUDA for all available GPUs (if installed and CUDA is present)

    Args:
        seed: Non-negative integer seed value.

    Raises:
        ValueError: If ``seed`` is negative.

    Note:
        PyTorch must be installed separately with the correct CUDA build
        for GPU seeding to take effect. See ``README.md`` for instructions.
        If PyTorch is absent, a warning is logged and the function returns
        normally — Python and NumPy seeds are still set.

    Example::

        set_seed(42)
    """
    if seed < 0:
        raise ValueError(
            f"Seed must be a non-negative integer, got {seed}. "
            f"Choose a value >= 0."
        )

    # Python built-in random
    random.seed(seed)
    logger.debug("Python random seed set to %d.", seed)

    # NumPy
    np.random.seed(seed)
    logger.debug("NumPy random seed set to %d.", seed)

    # PyTorch (optional — gracefully skip if not installed)
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        logger.debug("PyTorch CPU seed set to %d.", seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            logger.debug(
                "PyTorch CUDA seed set to %d across %d device(s).",
                seed,
                torch.cuda.device_count(),
            )
        else:
            logger.debug("CUDA not available; skipping torch.cuda seeding.")

    except ImportError:
        logger.warning(
            "PyTorch is not installed. Python and NumPy seeds were set, "
            "but PyTorch seeding was skipped. "
            "Install PyTorch with your GPU-specific CUDA build to enable GPU seeding."
        )
