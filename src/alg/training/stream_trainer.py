"""
Continual Stream Trainer for the Adaptive Learning Gate framework.

Supports two evaluation protocols on the identical continuous stream:
1. `StreamMode.CONVENTIONAL`:
   The standard continual learner — updates model on every incoming sample/batch
   regardless of noise or corruption (un-gated baseline).
2. `StreamMode.GATED`:
   The Adaptive Learning Gate (ALG) framework — selectively routes samples
   into LEARN, IGNORE, or DEFER, executes gradient steps only on LEARN samples,
   and periodically re-evaluates quarantined samples without infinite loops.
"""

from __future__ import annotations

import copy
import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alg.data.stream import StreamSample, StreamSimulator
from alg.evaluation.metrics import compute_classification_metrics
from alg.gating import AdaptiveLearningGate, DeferQueue, GateAction, GateDecision
from alg.signals import PrototypeMemory, SignalExtractor

logger = logging.getLogger(__name__)


class StreamMode(str, enum.Enum):
    """Operational mode for the stream trainer."""
    CONVENTIONAL = "conventional"  # Update on every stream sample
    GATED = "gated"                # Selective updates governed by the gate


@dataclass
class StreamStepRecord:
    """Audit record for a single stream batch or step."""
    step: int
    samples_seen: int
    learn_count: int
    ignore_count: int
    defer_count: int
    rechecked_learn_count: int
    loss: float | None
    val_accuracy: float | None = None


@dataclass
class StreamTrainingResult:
    """Comprehensive summary of continual stream training execution."""
    mode: StreamMode
    total_samples: int
    total_update_steps: int
    samples_learned: int
    samples_ignored: int
    samples_deferred: int
    rechecked_learned: int
    rechecked_ignored: int
    noise_ingested: int
    clean_learned: int
    final_test_accuracy: float
    final_test_f1: float
    accuracy_trajectory: list[dict[str, Any]] = field(default_factory=list)
    action_breakdown: dict[str, int] = field(default_factory=dict)
    compute_savings_pct: float = 0.0
    sample_savings_pct: float = 0.0


class StreamTrainer:
    """
    Orchestrates continuous stream training for both conventional and gated pipelines.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module | None = None,
        gate: AdaptiveLearningGate | None = None,
        signal_extractor: SignalExtractor | None = None,
        prototype_memory: PrototypeMemory | None = None,
        device: torch.device | str | None = None,
        use_amp: bool = True,
        recheck_interval: int = 5,
        recheck_batch_size: int = 32,
        max_defer_retries: int = 1,
    ) -> None:
        """
        Initialize the stream trainer.

        Args:
            model: PyTorch classification model.
            optimizer: Optimizer initialized with model parameters.
            criterion: Loss function (defaults to CrossEntropyLoss).
            gate: AdaptiveLearningGate coordinator (required for GATED mode).
            signal_extractor: SignalExtractor instance (required for GATED mode).
            prototype_memory: PrototypeMemory instance for updating centroids on LEARN.
            device: Computing device.
            use_amp: Whether to use mixed precision (FP16) on CUDA.
            recheck_interval: Number of stream batches between defer queue re-evaluations.
            recheck_batch_size: Maximum deferred samples to reconsider per recheck.
            max_defer_retries: Maximum times a sample can be re-deferred before retirement.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.gate = gate
        self.signal_extractor = signal_extractor
        self.prototype_memory = prototype_memory

        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.recheck_interval = recheck_interval
        self.recheck_batch_size = recheck_batch_size
        self.max_defer_retries = max_defer_retries

        # Retry tracking map: sample_id -> int retry_count
        self._retry_tracker: dict[str, int] = {}

    def _perform_gradient_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Execute forward pass, loss calculation, backward pass, and optimizer step.

        Returns:
            Computed loss value as a float.
        """
        if images.size(0) == 0:
            return 0.0

        # Guard against PyTorch BatchNorm size 1 crash in training mode
        is_single_sample = images.size(0) == 1
        if is_single_sample:
            images = torch.cat([images, images], dim=0)
            targets = torch.cat([targets, targets], dim=0)

        self.model.train()
        self.optimizer.zero_grad()

        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            logits = self.model(images)
            loss = self.criterion(logits, targets)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update prototype memory with newly learned representation
        if self.prototype_memory is not None and hasattr(self.model, "forward_features"):
            self.model.eval()
            with torch.no_grad():
                features = self.model.forward_features(images)
                self.prototype_memory.update(features, targets)
            self.model.train()

        return float(loss.item())

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
    ) -> dict[str, float]:
        """
        Evaluate current model on a held-out test or validation DataLoader.
        """
        self.model.eval()
        all_preds: list[int] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(images)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(targets.tolist())

        metrics = compute_classification_metrics(all_targets, all_preds)
        return {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        }

    def train_stream(
        self,
        simulator: StreamSimulator,
        batch_size: int = 64,
        mode: StreamMode = StreamMode.GATED,
        eval_loader: DataLoader | None = None,
        eval_interval: int = 500,
    ) -> StreamTrainingResult:
        """
        Run continual learning on the provided stream simulator.

        Args:
            simulator: StreamSimulator providing sequential stream batches.
            batch_size: Batch size for stream chunking.
            mode: StreamMode.CONVENTIONAL (train all) or StreamMode.GATED (ALG).
            eval_loader: Optional held-out DataLoader for tracking accuracy trajectory.
            eval_interval: Stream sample interval at which to evaluate accuracy.

        Returns:
            StreamTrainingResult with full empirical tracking data.
        """
        if mode == StreamMode.GATED and (self.gate is None or self.signal_extractor is None):
            raise ValueError("GATED stream training requires a gate and signal_extractor.")

        total_samples = len(simulator.samples)
        stream_loader = simulator.get_dataloader(batch_size=batch_size)

        samples_seen = 0
        total_update_steps = 0
        samples_learned = 0
        samples_ignored = 0
        samples_deferred = 0
        rechecked_learned = 0
        rechecked_ignored = 0
        noise_ingested = 0
        clean_learned = 0

        accuracy_trajectory: list[dict[str, Any]] = []

        logger.info(
            "Starting continual training: Mode=%s, Stream Size=%d samples, Batch Size=%d.",
            mode.value.upper(),
            total_samples,
            batch_size,
        )

        sample_offset = 0
        for batch_idx, (images, observed_labels) in enumerate(stream_loader):
            current_batch_size = images.size(0)
            raw_samples = simulator.samples[sample_offset : sample_offset + current_batch_size]
            sample_offset += current_batch_size
            samples_seen += current_batch_size

            if mode == StreamMode.CONVENTIONAL:
                # ── Conventional Baseline: Update unconditionally ────────────
                loss = self._perform_gradient_step(images, observed_labels)
                total_update_steps += 1
                samples_learned += current_batch_size

                # Count ground-truth noise ingested (for eval audit only)
                for s in raw_samples:
                    if s.metadata.is_noisy_label:
                        noise_ingested += 1
                    elif not s.metadata.is_corrupted_feature:
                        clean_learned += 1

            elif mode == StreamMode.GATED:
                # ── Gated Continual Training (ALG) ───────────────────────────
                # 1. Extract signals with current model
                scores = self.signal_extractor.extract_batch(images, observed_labels)

                learn_indices: list[int] = []
                for idx, sc in enumerate(scores):
                    s = raw_samples[idx]
                    decision = self.gate.evaluate_sample(
                        signals=sc,
                        sample=s,
                        sample_id=s.sample_id,
                        stream_index=s.stream_index,
                    )

                    if decision.action == GateAction.LEARN:
                        learn_indices.append(idx)
                        samples_learned += 1
                        if s.metadata.is_noisy_label:
                            noise_ingested += 1
                        elif not s.metadata.is_corrupted_feature:
                            clean_learned += 1

                    elif decision.action == GateAction.IGNORE:
                        samples_ignored += 1

                    elif decision.action == GateAction.DEFER:
                        samples_deferred += 1
                        self._retry_tracker[s.sample_id] = 0

                # 2. Perform gradient step only on LEARN samples
                if len(learn_indices) > 0:
                    l_imgs = images[learn_indices]
                    l_targets = observed_labels[learn_indices]
                    loss = self._perform_gradient_step(l_imgs, l_targets)
                    total_update_steps += 1

                # 3. Periodic Re-evaluation of Deferred Queue
                if (batch_idx + 1) % self.recheck_interval == 0 and len(self.gate.defer_queue) > 0:
                    rechecked_items = self.gate.defer_queue.pop_batch(self.recheck_batch_size)
                    if len(rechecked_items) > 0:
                        recheck_imgs = torch.stack([it[0].image for it in rechecked_items])
                        recheck_lbls = torch.tensor([it[0].observed_label for it in rechecked_items], dtype=torch.long)

                        recheck_scores = self.signal_extractor.extract_batch(recheck_imgs, recheck_lbls)

                        re_learn_idx: list[int] = []
                        for r_i, r_sc in enumerate(recheck_scores):
                            orig_sample, _ = rechecked_items[r_i]
                            r_dec = self.gate.policy.evaluate(
                                signals=r_sc,
                                sample_id=orig_sample.sample_id,
                                stream_index=orig_sample.stream_index,
                            )

                            if r_dec.action == GateAction.LEARN:
                                re_learn_idx.append(r_i)
                                rechecked_learned += 1
                                samples_learned += 1
                                if orig_sample.metadata.is_noisy_label:
                                    noise_ingested += 1
                                elif not orig_sample.metadata.is_corrupted_feature:
                                    clean_learned += 1

                            elif r_dec.action == GateAction.IGNORE:
                                rechecked_ignored += 1
                                samples_ignored += 1

                            elif r_dec.action == GateAction.DEFER:
                                # Check retry limit to prevent infinite loops
                                current_retries = self._retry_tracker.get(orig_sample.sample_id, 0)
                                if current_retries < self.max_defer_retries:
                                    self._retry_tracker[orig_sample.sample_id] = current_retries + 1
                                    self.gate.defer_queue.enqueue(orig_sample, r_dec)
                                else:
                                    # Retired after exceeding retries
                                    rechecked_ignored += 1
                                    samples_ignored += 1

                        if len(re_learn_idx) > 0:
                            rl_imgs = recheck_imgs[re_learn_idx]
                            rl_lbls = recheck_lbls[re_learn_idx]
                            self._perform_gradient_step(rl_imgs, rl_lbls)
                            total_update_steps += 1

            # ── Accuracy Tracking Checkpoint ─────────────────────────────────
            if eval_loader is not None and (samples_seen % eval_interval < current_batch_size or samples_seen >= total_samples):
                eval_metrics = self.evaluate_test_set(eval_loader)
                accuracy_trajectory.append({
                    "sample": samples_seen,
                    "test_acc": eval_metrics["accuracy"],
                    "test_f1": eval_metrics["macro_f1"],
                    "update_steps": total_update_steps,
                })
                logger.info(
                    "Stream Progress: %d/%d samples | Updates: %d | Test Acc: %.2f%%",
                    samples_seen,
                    total_samples,
                    total_update_steps,
                    eval_metrics["accuracy"] * 100,
                )

        # Final evaluation
        final_acc = 0.0
        final_f1 = 0.0
        if eval_loader is not None:
            final_metrics = self.evaluate_test_set(eval_loader)
            final_acc = final_metrics["accuracy"]
            final_f1 = final_metrics["macro_f1"]

        # Compute savings relative to updating on every stream batch
        total_possible_updates = len(stream_loader)
        compute_savings = (
            max(0.0, (1.0 - (total_update_steps / max(1, total_possible_updates)))) * 100
            if mode == StreamMode.GATED
            else 0.0
        )
        sample_savings = (
            max(0.0, (1.0 - (samples_learned / max(1, total_samples)))) * 100
            if mode == StreamMode.GATED
            else 0.0
        )

        return StreamTrainingResult(
            mode=mode,
            total_samples=total_samples,
            total_update_steps=total_update_steps,
            samples_learned=samples_learned,
            samples_ignored=samples_ignored,
            samples_deferred=samples_deferred,
            rechecked_learned=rechecked_learned,
            rechecked_ignored=rechecked_ignored,
            noise_ingested=noise_ingested,
            clean_learned=clean_learned,
            final_test_accuracy=final_acc,
            final_test_f1=final_f1,
            accuracy_trajectory=accuracy_trajectory,
            action_breakdown={
                "LEARN": samples_learned,
                "IGNORE": samples_ignored,
                "DEFER": samples_deferred,
            },
            compute_savings_pct=round(compute_savings, 2),
            sample_savings_pct=round(sample_savings, 2),
        )
