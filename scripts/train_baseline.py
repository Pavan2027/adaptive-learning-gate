#!/usr/bin/env python3
"""
Train the Conventional Baseline (Update-on-Every-Sample).

Usage::

    # Standard CIFAR-10 baseline run with GPU:
    python scripts/train_baseline.py

    # Quick smoke test on CPU:
    python scripts/train_baseline.py --smoke-test --device cpu

    # Run baseline on Fashion-MNIST:
    python scripts/train_baseline.py --dataset fashion_mnist

    # Custom epochs and architecture:
    python scripts/train_baseline.py --architecture cnn --epochs 10

Results and checkpoints are automatically saved to `results/` and `checkpoints/`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn

from alg.config import load_config
from alg.data.datasets import get_dataloaders
from alg.evaluation.metrics import save_metrics_summary
from alg.models import build_model
from alg.training.baseline import BaselineTrainer
from alg.utils.seeding import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the conventional baseline (update-on-every-sample) for ALG."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "fashion_mnist", "mnist"],
        help="Override dataset name.",
    )
    parser.add_argument(
        "--architecture",
        choices=["cnn", "mlp"],
        help="Override model architecture.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override optimizer learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Force execution device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick CPU smoke-test (1 epoch, 2 batches).",
    )
    parser.add_argument(
        "--output",
        help="Path for saving results JSON file. Defaults to results/baseline_<dataset>_<arch>.json",
    )
    args = parser.parse_args()

    # 1. Load and update configuration
    cfg = load_config(args.config)
    dataset_name = args.dataset or cfg.dataset.name
    architecture = args.architecture or cfg.model.architecture
    epochs = args.epochs or cfg.epochs
    batch_size = args.batch_size or cfg.batch_size
    lr = args.lr or cfg.optimizer.lr
    seed = args.seed if args.seed is not None else cfg.seed

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # 2. Set deterministic seed
    set_seed(seed)

    print("=" * 64)
    print("   Adaptive Learning Gate — Conventional Baseline Training")
    print("=" * 64)
    print(f"  Dataset      : {dataset_name}")
    print(f"  Architecture : {architecture.upper()}")
    print(f"  Device       : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Epochs       : {1 if args.smoke_test else epochs}")
    print(f"  Batch Size   : {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Seed         : {seed}")
    print(f"  Smoke Test   : {args.smoke_test}")
    print("=" * 64)

    # 3. Load Datasets & DataLoaders
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        dataset_name=dataset_name,
        root=cfg.dataset.root,
        batch_size=batch_size,
        val_fraction=0.1,
        seed=seed,
        download=cfg.dataset.download,
    )

    # 4. Build Model
    model = build_model(
        architecture=architecture,
        num_classes=metadata.num_classes,
        in_channels=metadata.in_channels,
        image_size=metadata.image_size,
        hidden_dims=cfg.model.hidden_dims,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model '%s' initialized with %d trainable parameters.", architecture.upper(), num_params)

    # 5. Build Optimizer & Loss
    if cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=cfg.optimizer.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=cfg.optimizer.weight_decay
        )

    criterion = nn.CrossEntropyLoss()

    # 6. Run Baseline Training
    trainer = BaselineTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_classes=metadata.num_classes,
    )

    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        checkpoint_dir=cfg.output.checkpoint_dir,
        smoke_test=args.smoke_test,
        config_snapshot=cfg.model_dump(),
    )

    # 7. Attach Run Metadata & Save
    results["metadata"] = {
        "dataset": dataset_name,
        "architecture": architecture,
        "seed": seed,
        "num_params": num_params,
        "config": cfg.model_dump(),
    }

    out_path = args.output or f"{cfg.output.results_dir}/baseline_{dataset_name}_{architecture}_results.json"
    save_metrics_summary(results, out_path)

    # 8. Display Final Summary
    test_m = results["test_metrics"]
    print("\n" + "=" * 64)
    print("   Baseline Training Finished Successfully!")
    print("=" * 64)
    print(f"  Best Validation Epoch : {results['best_epoch']}")
    print(f"  Best Validation Acc   : {results['best_val_acc'] * 100:.2f}%")
    print(f"  Held-out Test Acc     : {test_m['accuracy'] * 100:.2f}%")
    print(f"  Held-out Test Macro-F1: {test_m['macro_f1'] * 100:.2f}%")
    print(f"  Held-out Test Loss    : {test_m['loss']:.4f}")
    print(f"  Total Wall Time       : {results['total_time_seconds']}s")
    print(f"  Results saved to      : {out_path}")
    print(f"  Checkpoint saved to   : {results['checkpoint_path']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
