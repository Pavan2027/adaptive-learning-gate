"""
Conventional Training Baseline (Update-on-Every-Sample).

Implements the standard supervised learning benchmark where every training sample
triggers a model parameter update in each epoch.

Features:
- Epoch-by-epoch training and validation tracking.
- Best validation model checkpointing.
- Final evaluation on strictly held-out test data.
- Machine-readable result logging.
- Smoke-test mode for rapid CPU verification.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alg.evaluation.metrics import evaluate_model

logger = logging.getLogger(__name__)


def save_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    val_metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Save model weights, optimizer state, and training metadata to disk.

    Args:
        filepath: Target .pt file path.
        model: PyTorch model.
        optimizer: PyTorch optimizer (optional).
        epoch: Epoch number.
        val_metrics: Validation metrics at this checkpoint.
        config: Configuration dictionary snapshot.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "val_metrics": val_metrics or {},
        "config": config or {},
    }
    torch.save(state, path)
    logger.debug("Checkpoint saved to '%s'.", path)


def load_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """
    Restore model weights and optimizer state from a checkpoint file.

    Args:
        filepath: Path to checkpoint .pt file.
        model: PyTorch model to populate.
        optimizer: PyTorch optimizer to populate (optional).
        device: Device to map tensors onto.

    Returns:
        Checkpoint dictionary containing metadata ('epoch', 'val_metrics', 'config').
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{path}'.")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info("Restored checkpoint from '%s' (epoch %d).", path, checkpoint.get("epoch", 0))
    return checkpoint


class BaselineTrainer:
    """
    Standard supervised trainer executing the update-on-every-sample baseline.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module | None = None,
        device: torch.device | str | None = None,
        num_classes: int = 10,
        use_amp: bool = True,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.num_classes = num_classes

        # Enable CUDA Tensor Core optimizations and mixed precision (FP16)
        self.use_amp = use_amp and (self.device.type == "cuda")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_epoch(
        self,
        dataloader: DataLoader,
        smoke_test_batches: int | None = None,
    ) -> dict[str, float]:
        """
        Train for a single epoch across all batches in the dataloader.

        Returns:
            Dictionary with 'train_loss', 'train_acc', and 'step_count'.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct_samples = 0
        step_count = 0

        for batch_idx, (images, targets) in enumerate(dataloader):
            if smoke_test_batches is not None and batch_idx >= smoke_test_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            step_count += 1

            preds = torch.argmax(logits, dim=1)
            correct_samples += int((preds == targets).sum().item())

            if smoke_test_batches is None and (batch_idx + 1) % 50 == 0:
                logger.info("  Step %d/%d | Batch Loss: %.4f", batch_idx + 1, len(dataloader), loss.item())

        avg_loss = float(total_loss / total_samples) if total_samples > 0 else 0.0
        train_acc = float(correct_samples / total_samples) if total_samples > 0 else 0.0

        return {
            "train_loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 4),
            "step_count": step_count,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 5,
        checkpoint_dir: str | Path = "checkpoints/",
        smoke_test: bool = False,
        config_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute full baseline training loop over specified epochs.

        Args:
            train_loader: Training set loader.
            val_loader: Validation set loader (used for model selection).
            test_loader: Held-out test loader (evaluated once on best model).
            epochs: Number of epochs to train.
            checkpoint_dir: Directory where checkpoints will be stored.
            smoke_test: If True, limit each epoch to 2 batches for fast testing.
            config_snapshot: Optional dictionary of run configuration to save.

        Returns:
            Complete results dictionary with per-epoch logs and final test metrics.
        """
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = ckpt_dir / "best_model.pt"

        max_batches = 2 if smoke_test else None
        num_epochs = 1 if smoke_test else epochs

        logger.info(
            "Starting baseline training for %d epochs on %s (smoke_test=%s).",
            num_epochs,
            self.device,
            smoke_test,
        )

        history: list[dict[str, Any]] = []
        best_val_acc = -1.0
        best_epoch = 0
        total_steps = 0
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train epoch
            train_metrics = self.train_epoch(train_loader, smoke_test_batches=max_batches)
            total_steps += int(train_metrics["step_count"])

            # Evaluate on validation set
            val_metrics = evaluate_model(
                self.model,
                val_loader,
                self.criterion,
                self.device,
                num_classes=self.num_classes,
                max_batches=max_batches,
            )

            epoch_time = time.time() - epoch_start
            epoch_log = {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "train_acc": train_metrics["train_acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "epoch_time_seconds": round(epoch_time, 2),
            }
            history.append(epoch_log)

            logger.info(
                "Epoch %d/%d | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f F1: %.4f (%.1fs)",
                epoch,
                num_epochs,
                train_metrics["train_loss"],
                train_metrics["train_acc"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                epoch_time,
            )

            # Checkpoint best validation model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                save_checkpoint(
                    best_ckpt_path,
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    val_metrics=val_metrics,
                    config=config_snapshot,
                )

        total_wall_clock = time.time() - start_time
        logger.info(
            "Training completed in %.2fs. Best Val Acc: %.4f (Epoch %d).",
            total_wall_clock,
            best_val_acc,
            best_epoch,
        )

        # Restore best model for final evaluation on strictly held-out test data
        if best_ckpt_path.exists():
            load_checkpoint(best_ckpt_path, self.model, device=self.device)

        test_metrics = evaluate_model(
            self.model,
            test_loader,
            self.criterion,
            self.device,
            num_classes=self.num_classes,
            max_batches=max_batches,
        )

        logger.info(
            "Final Test Metrics (Held-out): Acc: %.4f | F1: %.4f | Loss: %.4f",
            test_metrics["accuracy"],
            test_metrics["macro_f1"],
            test_metrics["loss"],
        )

        return {
            "baseline_type": "update_on_every_sample",
            "device": str(self.device),
            "epochs_trained": num_epochs,
            "total_steps": total_steps,
            "total_time_seconds": round(total_wall_clock, 2),
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_metrics": test_metrics,
            "history": history,
            "checkpoint_path": str(best_ckpt_path),
        }
