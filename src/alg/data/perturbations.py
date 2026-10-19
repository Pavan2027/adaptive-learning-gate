"""
Controlled perturbation transformations for stream simulation.

Supports:
- Label Noise: Symmetric random class flipping (model observes wrong label).
- Feature Corruption: Gaussian pixel noise, cutout (square masking), or blur.
- Redundancy: Insertion of duplicate/repeated samples.
- Distribution Shift: Partitioning stream into segments with varying class priors.

All transformations take an explicit random generator to guarantee full reproducibility.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def apply_label_noise(
    true_label: int,
    num_classes: int,
    noise_rate: float,
    rng: np.random.Generator,
) -> tuple[int, bool]:
    """
    Symmetric random label flipping.

    With probability `noise_rate`, the label is corrupted to any class in
    {0, ..., num_classes - 1} excluding the true label.

    Args:
        true_label: Ground truth integer class.
        num_classes: Total number of classes.
        noise_rate: Probability of corruption in [0.0, 1.0].
        rng: Deterministic NumPy Generator.

    Returns:
        Tuple of (observed_label, was_corrupted).
    """
    if noise_rate <= 0.0 or rng.random() > noise_rate:
        return true_label, False

    candidates = [c for c in range(num_classes) if c != true_label]
    corrupted_label = int(rng.choice(candidates))
    return corrupted_label, True


def apply_feature_corruption(
    image: torch.Tensor,
    corruption_rate: float,
    corruption_type: str = "gaussian",
    severity: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, bool, str]:
    """
    Controlled perturbation applied to an image tensor.

    Args:
        image: Tensor of shape (C, H, W).
        corruption_rate: Probability of applying corruption.
        corruption_type: 'gaussian', 'cutout', or 'blur'.
        severity: Strength of corruption in (0.0, 1.0].
        rng: Deterministic NumPy Generator.

    Returns:
        Tuple of (output_tensor, was_corrupted, type_applied).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if corruption_rate <= 0.0 or rng.random() > corruption_rate:
        return image.clone(), False, "none"

    corrupted = image.clone()
    c, h, w = image.shape

    if corruption_type == "gaussian":
        # Add zero-mean Gaussian noise scaled by severity
        std = float(severity * 0.75)
        noise = torch.tensor(rng.normal(0.0, std, size=image.shape), dtype=image.dtype)
        corrupted = corrupted + noise

    elif corruption_type == "cutout":
        # Mask a square region with zeros
        mask_size = max(4, int(min(h, w) * severity * 0.6))
        top = int(rng.integers(0, max(1, h - mask_size)))
        left = int(rng.integers(0, max(1, w - mask_size)))
        corrupted[:, top : top + mask_size, left : left + mask_size] = 0.0

    elif corruption_type == "blur":
        # 3x3 uniform box filter smoothing
        kernel = torch.ones((c, 1, 3, 3), dtype=image.dtype) / 9.0
        # Expand to batch dimension for conv2d
        x = corrupted.unsqueeze(0)
        blurred = F.conv2d(x, kernel, padding=1, groups=c)
        corrupted = blurred.squeeze(0)

    else:
        raise ValueError(
            f"Unknown corruption_type '{corruption_type}'. Supported: ['gaussian', 'cutout', 'blur']."
        )

    return corrupted, True, corruption_type
