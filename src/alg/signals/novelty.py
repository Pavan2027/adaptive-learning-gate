"""
Representation-based novelty signal extraction using bounded prototype memory.

Definition:
Measures the semantic distance between the current sample's penultimate feature
representation z and the model's memory of previously observed class prototypes {mu_c}.

Mathematical Formulation:
1. Feature normalization:
   z_hat = z / ||z||_2
2. Minimum Cosine Distance to known prototypes:
   d_min = min_c (1.0 - <z_hat, mu_hat_c>) / 2.0
   - Scale: strictly in [0.0, 1.0]
   - d_min = 0.0: identical to a known class prototype (redundant / highly familiar)
   - d_min = 1.0: orthogonal or opposite to all known representations (highly novel)

Update Policy:
Class centroids are updated online via Exponential Moving Average (EMA) with
configurable momentum (default 0.95), using constant O(C * D) bounded memory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PrototypeMemory:
    """
    Bounded, deterministic class prototype memory for novelty estimation.
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 256,
        momentum: float = 0.95,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.device = torch.device(device)

        # Centroid matrix of shape (num_classes, feature_dim)
        self.prototypes = torch.zeros(
            (num_classes, feature_dim), dtype=torch.float32, device=self.device
        )
        # Track whether each prototype has received at least one sample
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=self.device)

    def to(self, device: torch.device | str) -> "PrototypeMemory":
        """Move memory tensors to target device."""
        self.device = torch.device(device)
        self.prototypes = self.prototypes.to(self.device)
        self.initialized = self.initialized.to(self.device)
        return self

    def compute_novelty(
        self,
        features: torch.Tensor,
        observed_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute normalized novelty score for feature embeddings.

        Args:
            features: Tensor of shape (B, feature_dim) or (feature_dim,).
            observed_labels: Optional observed labels (if computing class-conditional novelty).

        Returns:
            Tensor of shape (B,) containing novelty scores in [0.0, 1.0].
        """
        if features.ndim == 1:
            features = features.unsqueeze(0)

        features = features.to(self.device)
        batch_size = features.size(0)

        # If memory is completely uninitialized, all samples are novel
        if not self.initialized.any():
            return torch.ones(batch_size, dtype=torch.float32, device=self.device)

        # Normalize feature embeddings
        z_norm = F.normalize(features, p=2, dim=-1)

        # Use only initialized prototypes
        active_prototypes = self.prototypes[self.initialized]
        active_norm = F.normalize(active_prototypes, p=2, dim=-1)

        # Compute cosine similarity matrix: (B, num_active)
        cos_sim = torch.mm(z_norm, active_norm.t())

        # Cosine distance: (1.0 - cos_sim) / 2.0 mapped to [0.0, 1.0]
        cos_dist = (1.0 - cos_sim) / 2.0

        # Novelty is the distance to the nearest existing prototype
        min_dist, _ = torch.min(cos_dist, dim=-1)

        min_dist = torch.nan_to_num(min_dist, nan=1.0, posinf=1.0, neginf=0.0)
        return torch.clamp(min_dist, min=0.0, max=1.0)

    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Update class prototypes with accepted training samples.

        Args:
            features: Tensor of shape (B, feature_dim).
            labels: Tensor of integer class labels of shape (B,).
        """
        features = features.detach().to(self.device)
        labels = labels.to(self.device)

        for c in range(self.num_classes):
            mask = labels == c
            if not mask.any():
                continue

            class_mean = features[mask].mean(dim=0)
            if not self.initialized[c]:
                self.prototypes[c] = class_mean
                self.initialized[c] = True
            else:
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c] + (1.0 - self.momentum) * class_mean
                )
