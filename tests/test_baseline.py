"""
Tests for alg.training.baseline.

Covers:
- Checkpoint save and load round-trip (model weight consistency).
- BaselineTrainer train_epoch functionality on CPU.
- BaselineTrainer fit smoke test and final held-out evaluation.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from alg.models import CompactCNN
from alg.training.baseline import BaselineTrainer, load_checkpoint, save_checkpoint


@pytest.fixture
def dummy_data() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create tiny synthetic data loaders for fast CPU unit testing."""
    # 16 samples per split, 3 channels, 32x32 images, 4 classes
    x_train = torch.randn(16, 3, 32, 32)
    y_train = torch.randint(0, 4, (16,))
    x_val = torch.randn(8, 3, 32, 32)
    y_val = torch.randint(0, 4, (8,))
    x_test = torch.randn(8, 3, 32, 32)
    y_test = torch.randint(0, 4, (8,))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=4)
    return train_loader, val_loader, test_loader


class TestCheckpointRoundtrip:
    def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
        model1 = CompactCNN(in_channels=3, num_classes=4)
        optimizer = torch.optim.Adam(model1.parameters(), lr=1e-3)
        ckpt_path = tmp_path / "model.pt"

        save_checkpoint(
            ckpt_path,
            model1,
            optimizer,
            epoch=3,
            val_metrics={"accuracy": 0.85},
            config={"seed": 42},
        )

        assert ckpt_path.exists()

        # Instantiate fresh model and restore
        model2 = CompactCNN(in_channels=3, num_classes=4)
        loaded = load_checkpoint(ckpt_path, model2, device="cpu")

        assert loaded["epoch"] == 3
        assert loaded["val_metrics"]["accuracy"] == 0.85

        # Verify weights are identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)


class TestBaselineTrainer:
    def test_train_epoch_on_cpu(
        self, dummy_data: tuple[DataLoader, DataLoader, DataLoader]
    ) -> None:
        train_loader, _, _ = dummy_data
        model = CompactCNN(in_channels=3, num_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = BaselineTrainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            num_classes=4,
        )

        metrics = trainer.train_epoch(train_loader)
        assert "train_loss" in metrics
        assert "train_acc" in metrics
        assert metrics["step_count"] == 4

    def test_fit_smoke_test(
        self, dummy_data: tuple[DataLoader, DataLoader, DataLoader], tmp_path: Path
    ) -> None:
        train_loader, val_loader, test_loader = dummy_data
        model = CompactCNN(in_channels=3, num_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = BaselineTrainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            num_classes=4,
        )

        results = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=1,
            checkpoint_dir=tmp_path,
            smoke_test=True,
        )

        assert results["baseline_type"] == "update_on_every_sample"
        assert "test_metrics" in results
        assert "accuracy" in results["test_metrics"]
        assert len(results["history"]) == 1
        assert (tmp_path / "best_model.pt").exists()
