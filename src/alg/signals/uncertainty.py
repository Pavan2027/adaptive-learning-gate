"""
Predictive uncertainty signal extraction.

Methods:
1. Normalized Predictive Entropy:
   H(p) = -sum_c (p_c * log2(p_c + eps)) / log2(C)
   - Scale: [0.0, 1.0]
   - H = 0.0 when model assigns 1.0 probability to a single class (maximum confidence).
   - H = 1.0 when model assigns uniform 1/C probability to all classes (maximum uncertainty).

2. Margin Uncertainty:
   M(p) = 1.0 - (p_top1 - p_top2)
   - Scale: [0.0, 1.0]
   - M = 0.0 when top prediction dominates completely.
   - M = 1.0 when top two predictions are tied.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def compute_predictive_entropy(
    logits: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute normalized predictive Shannon entropy across class logits.

    Args:
        logits: Unnormalized logits tensor of shape (B, C) or (C,).
        epsilon: Small numerical stability constant to avoid log(0).

    Returns:
        Tensor of shape (B,) or scalar containing normalized entropy in [0.0, 1.0].
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    num_classes = logits.size(-1)
    if num_classes <= 1:
        return torch.zeros(logits.size(0), dtype=torch.float32, device=logits.device)

    # Softmax probabilities
    probs = F.softmax(logits, dim=-1)

    # Shannon entropy with base-2 logarithm: H = -sum(p * log2(p))
    log_probs = torch.log2(probs + epsilon)
    raw_entropy = -torch.sum(probs * log_probs, dim=-1)

    # Theoretical maximum entropy for uniform distribution is log2(C)
    max_entropy = math.log2(float(num_classes))
    normalized_entropy = raw_entropy / max_entropy

    # Handle any potential non-finite values and clamp strictly to [0.0, 1.0]
    normalized_entropy = torch.nan_to_num(normalized_entropy, nan=1.0, posinf=1.0, neginf=0.0)
    return torch.clamp(normalized_entropy, min=0.0, max=1.0)


def compute_margin_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute margin-based uncertainty (1.0 minus difference between top two probabilities).

    Args:
        logits: Unnormalized logits tensor of shape (B, C) or (C,).

    Returns:
        Tensor of shape (B,) containing margin uncertainty in [0.0, 1.0].
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    num_classes = logits.size(-1)
    if num_classes <= 1:
        return torch.zeros(logits.size(0), dtype=torch.float32, device=logits.device)

    probs = F.softmax(logits, dim=-1)
    top2_values, _ = torch.topk(probs, k=2, dim=-1)

    # Margin = top1 - top2 (large margin = high confidence = low uncertainty)
    margin = top2_values[:, 0] - top2_values[:, 1]
    margin_uncertainty = 1.0 - margin

    margin_uncertainty = torch.nan_to_num(margin_uncertainty, nan=1.0, posinf=1.0, neginf=0.0)
    return torch.clamp(margin_uncertainty, min=0.0, max=1.0)


def compute_predictive_uncertainty(
    logits: torch.Tensor,
    method: str = "entropy",
) -> torch.Tensor:
    """
    Dispatcher to compute predictive uncertainty via chosen method.

    Args:
        logits: Tensor of unnormalized model logits.
        method: 'entropy' or 'margin'.

    Returns:
        Tensor of normalized uncertainty scores in [0.0, 1.0].
    """
    if method == "entropy":
        return compute_predictive_entropy(logits)
    elif method == "margin":
        return compute_margin_uncertainty(logits)
    else:
        raise ValueError(f"Unknown uncertainty method '{method}'. Supported: ['entropy', 'margin'].")
