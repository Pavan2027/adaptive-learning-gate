"""
Unified Signal Extractor for computing per-sample learning signals.

Executes a single forward pass per batch in evaluation mode (`torch.no_grad()`)
to extract:
1. Predictive Uncertainty (Normalized Shannon entropy)
2. Semantic Novelty (Distance to known class prototypes)
3. Sample Reliability (Feature sanity & predictive consistency)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from alg.signals.novelty import PrototypeMemory
from alg.signals.reliability import compute_feature_integrity, compute_label_consistency, compute_reliability
from alg.signals.types import SignalScores
from alg.signals.uncertainty import compute_predictive_uncertainty


class SignalExtractor:
    """
    Computes all learning signals in a single forward pass without backprop overhead.
    """

    def __init__(
        self,
        model: nn.Module,
        prototype_memory: PrototypeMemory | None = None,
        uncertainty_method: str = "entropy",
        device: torch.device | str | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.uncertainty_method = uncertainty_method
        self.memory = prototype_memory
        if self.memory is not None:
            self.memory.to(self.device)

    def extract_batch(
        self,
        images: torch.Tensor,
        observed_labels: torch.Tensor,
    ) -> list[SignalScores]:
        """
        Extract normalized signal scores for a batch of images and observed labels.

        Args:
            images: Tensor of shape (B, C, H, W).
            observed_labels: Tensor of shape (B,).

        Returns:
            List of SignalScores objects for each sample in the batch.
        """
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        observed_labels = observed_labels.to(self.device, non_blocking=True)

        with torch.no_grad():
            # 1. Feature extraction pass
            if hasattr(self.model, "forward_features"):
                features = self.model.forward_features(images)
                if hasattr(self.model, "classifier"):
                    logits = self.model.classifier(features)
                else:
                    logits = self.model(images)
            else:
                features = None
                logits = self.model(images)

            # 2. Predictive Uncertainty
            uncertainties = compute_predictive_uncertainty(logits, method=self.uncertainty_method)

            # 3. Novelty
            if self.memory is not None and features is not None:
                novelties = self.memory.compute_novelty(features, observed_labels)
            else:
                # Default novelty score if no feature memory provided
                novelties = torch.zeros(images.size(0), dtype=torch.float32, device=self.device)

            # 4. Reliability
            reliabilities = compute_reliability(images, logits, observed_labels)
            r_feat = compute_feature_integrity(images)
            r_label = compute_label_consistency(logits, observed_labels)

        # Convert to CPU list of SignalScores
        batch_size = images.size(0)
        u_list = uncertainties.cpu().tolist()
        n_list = novelties.cpu().tolist()
        r_list = reliabilities.cpu().tolist()
        rf_list = r_feat.cpu().tolist()
        rl_list = r_label.cpu().tolist()

        scores = []
        for i in range(batch_size):
            scores.append(
                SignalScores(
                    uncertainty=float(u_list[i]),
                    novelty=float(n_list[i]),
                    reliability=float(r_list[i]),
                    raw_metrics={
                        "feature_integrity": float(rf_list[i]),
                        "label_consistency": float(rl_list[i]),
                    },
                )
            )

        return scores
