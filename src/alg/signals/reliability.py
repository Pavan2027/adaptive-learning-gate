"""
Reliability signal extraction using observable input integrity and predictive consistency.

Signals:
1. Feature Integrity (R_feat):
   Checks for:
   - Non-finite pixel values (NaN / Inf -> R_feat = 0.0).
   - Abnormally high spatial variance / high-frequency noise indicative of corruptions.
   - Scale: strictly in [0.0, 1.0].

2. Observed Label Consistency (R_label):
   Evaluates agreement between the model's predictive distribution and the incoming observed label:
   R_label = p(y_obs | x) / (max_c p(c | x) + eps)
   - When observed label matches top prediction: R_label = 1.0.
   - When model is highly confident in class A, but observed label is class B: R_label << 1.0.
   - Scale: strictly in [0.0, 1.0].

Composite Reliability:
R = R_feat * R_label in [0.0, 1.0].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_feature_integrity(images: torch.Tensor) -> torch.Tensor:
    """
    Compute input feature integrity scores.

    Args:
        images: Image tensor of shape (B, C, H, W).

    Returns:
        Tensor of shape (B,) containing integrity scores in [0.0, 1.0].
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)

    batch_size = images.size(0)
    device = images.device

    # 1. Finite values check
    is_finite = torch.isfinite(images).view(batch_size, -1).all(dim=-1)
    if not is_finite.all():
        scores = torch.zeros(batch_size, dtype=torch.float32, device=device)
        return torch.where(is_finite, torch.ones_like(scores), scores)

    # 2. Local spatial variance (detects severe high-frequency corruptions)
    # Natural normalized images typically have std in [0.2, 1.2]
    # Heavy Gaussian or salt-and-pepper noise produces extreme variance
    pixel_stds = images.view(batch_size, -1).std(dim=-1)

    # Soft sigmoid penalty for extreme variance (> 2.5 standard deviations)
    # score = 1.0 / (1.0 + exp(pixel_std - threshold))
    extreme_penalty = torch.sigmoid((pixel_stds - 2.5) * 4.0)
    integrity = 1.0 - extreme_penalty

    integrity = torch.nan_to_num(integrity, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(integrity, min=0.0, max=1.0)


def compute_label_consistency(
    logits: torch.Tensor,
    observed_labels: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute predictive consistency between model class distribution and observed label.

    Formulation:
        R_label = p(y_obs | x) / (max_c p(c | x) + eps)

    Args:
        logits: Tensor of unnormalized model logits of shape (B, C).
        observed_labels: Tensor of observed integer targets of shape (B,).
        epsilon: Small numerical stability constant.

    Returns:
        Tensor of shape (B,) containing consistency scores in [0.0, 1.0].
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if observed_labels.ndim == 0:
        observed_labels = observed_labels.unsqueeze(0)

    probs = F.softmax(logits, dim=-1)

    # Gather model's assigned probability to the observed label
    obs_prob = probs.gather(1, observed_labels.unsqueeze(1)).squeeze(1)

    # Maximum model probability across all classes
    max_prob, _ = torch.max(probs, dim=-1)

    # Relative consistency ratio
    consistency = obs_prob / (max_prob + epsilon)

    consistency = torch.nan_to_num(consistency, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(consistency, min=0.0, max=1.0)


def compute_reliability(
    images: torch.Tensor,
    logits: torch.Tensor,
    observed_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute composite sample reliability score combining feature sanity and label consistency.

    Args:
        images: Input images tensor of shape (B, C, H, W).
        logits: Model predicted logits of shape (B, C).
        observed_labels: Target labels of shape (B,).

    Returns:
        Tensor of shape (B,) with reliability scores in [0.0, 1.0].
    """
    r_feat = compute_feature_integrity(images)
    r_label = compute_label_consistency(logits, observed_labels)

    # Composite reliability is the product of feature integrity and label consistency
    reliability = r_feat * r_label
    return torch.clamp(reliability, min=0.0, max=1.0)
