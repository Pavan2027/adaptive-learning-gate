#!/usr/bin/env python3
"""
Execute Continual Stream Training experiments (Conventional vs. Gated ALG).

Allows running:
1. `conventional`: Continuous training on every stream sample (un-gated baseline).
2. `gated`: Adaptive Learning Gate filtering (selective continual training).
3. `compare`: Side-by-side comparison on the identical stream manifest and model checkpoint.

Usage::

    python scripts/run_stream.py --mode compare --stream-size 2000
    python scripts/run_stream.py --mode gated --stream-size 2000 --label-noise 0.20
    python scripts/run_stream.py --export research/data/stream_comparison.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from alg.config import load_config
from alg.data.datasets import get_dataset_metadata, load_raw_dataset
from alg.data.stream import StreamSimulator
from alg.gating import AdaptiveLearningGate
from alg.models import build_model
from alg.signals import PrototypeMemory, SignalExtractor
from alg.training.baseline import load_checkpoint
from alg.training.stream_trainer import StreamMode, StreamTrainer, StreamTrainingResult
from alg.utils.seeding import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stream_runner")


def _build_test_loader(dataset_name: str, root: str, batch_size: int = 128) -> DataLoader:
    """Build held-out test dataloader."""
    meta = get_dataset_metadata(dataset_name)
    in_channels = meta.in_channels
    if in_channels == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])

    raw_test = load_raw_dataset(dataset_name, root=root, train=False, download=True, transform=transform)
    return DataLoader(raw_test, batch_size=batch_size, shuffle=False)


def run_experiment(
    mode: str,
    stream_size: int,
    batch_size: int,
    label_noise: float,
    feature_corruption: float,
    checkpoint_path: str,
    config_path: str,
    export_path: str,
) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    print("=" * 68)
    print("   Adaptive Learning Gate — Continual Stream Training Runner")
    print("=" * 68)
    print(f"  Mode                 : {mode.upper()}")
    print(f"  Model Checkpoint     : {checkpoint_path}")
    print(f"  Stream Size          : {stream_size:,} samples")
    print(f"  Batch Size           : {batch_size}")
    print(f"  Simulated Noise      : {label_noise * 100:.1f}% Label, {feature_corruption * 100:.1f}% Feat")
    print(f"  Device               : {device}")
    print("=" * 68)

    # 1. Load stream dataset
    meta = get_dataset_metadata(cfg.dataset.name)
    in_channels = meta.in_channels
    if in_channels == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])

    raw_train = load_raw_dataset(cfg.dataset.name, root=cfg.dataset.root, train=True, download=cfg.dataset.download, transform=transform)
    test_loader = _build_test_loader(cfg.dataset.name, cfg.dataset.root, batch_size=128)

    # 2. Build reproducible perturbed stream
    simulator = StreamSimulator(
        base_dataset=raw_train,
        stream_size=stream_size,
        label_noise_rate=label_noise,
        feature_corruption_rate=feature_corruption,
        seed=cfg.seed,
    )

    # Warmup memory dataset
    def _warmup_memory(model: torch.nn.Module, memory: PrototypeMemory) -> None:
        with torch.no_grad():
            w_loader = DataLoader(raw_train, batch_size=128, shuffle=True)
            for b_idx, (imgs, lbls) in enumerate(w_loader):
                if b_idx >= 5:
                    break
                feats = model.forward_features(imgs.to(device))
                memory.update(feats, lbls.to(device))

    results: dict[str, Any] = {}

    modes_to_run = ["conventional", "gated"] if mode == "compare" else [mode]

    for run_mode in modes_to_run:
        print(f"\n>>> Running Continual Training: [{run_mode.upper()}] Mode")
        # Load fresh model from baseline checkpoint
        model = build_model(architecture=cfg.model.architecture, num_classes=10, in_channels=in_channels)
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            load_checkpoint(ckpt, model, device=device)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=cfg.optimizer.weight_decay)

        if run_mode == "conventional":
            trainer = StreamTrainer(
                model=model,
                optimizer=optimizer,
                device=device,
                use_amp=True,
            )
            res = trainer.train_stream(
                simulator=simulator,
                batch_size=batch_size,
                mode=StreamMode.CONVENTIONAL,
                eval_loader=test_loader,
                eval_interval=max(200, stream_size // 4),
            )
        else:
            memory = PrototypeMemory(num_classes=10, feature_dim=256, device=device)
            _warmup_memory(model, memory)

            extractor = SignalExtractor(model=model, prototype_memory=memory, device=device)
            gate = AdaptiveLearningGate(
                learn_threshold=cfg.gate.learn_threshold,
                ignore_threshold=cfg.gate.ignore_threshold,
                defer_queue_capacity=cfg.gate.defer_queue_capacity,
            )
            trainer = StreamTrainer(
                model=model,
                optimizer=optimizer,
                gate=gate,
                signal_extractor=extractor,
                prototype_memory=memory,
                device=device,
                use_amp=True,
                recheck_interval=5,
                recheck_batch_size=32,
            )
            res = trainer.train_stream(
                simulator=simulator,
                batch_size=batch_size,
                mode=StreamMode.GATED,
                eval_loader=test_loader,
                eval_interval=max(200, stream_size // 4),
            )

        results[run_mode] = {
            "total_samples": res.total_samples,
            "total_updates": res.total_update_steps,
            "samples_learned": res.samples_learned,
            "samples_ignored": res.samples_ignored,
            "samples_deferred": res.samples_deferred,
            "noise_ingested": res.noise_ingested,
            "clean_learned": res.clean_learned,
            "final_test_accuracy": round(res.final_test_accuracy * 100, 2),
            "final_test_f1": round(res.final_test_f1 * 100, 2),
            "compute_savings_pct": res.compute_savings_pct,
            "trajectory": res.accuracy_trajectory,
        }

    # 3. Print Comparison Table
    print("\n" + "=" * 68)
    print("   Continual Stream Training — Empirical Performance Summary")
    print("=" * 68)
    print(f"  Metric                        | Conventional (Un-gated) | Gated ALG (Ours)")
    print("  " + "-" * 66)

    conv = results.get("conventional")
    gtd = results.get("gated")

    if conv and gtd:
        print(f"  Final Test Accuracy           | {conv['final_test_accuracy']:>18.2f}% | {gtd['final_test_accuracy']:>14.2f}%")
        print(f"  Final Macro-F1                | {conv['final_test_f1']:>18.2f}% | {gtd['final_test_f1']:>14.2f}%")
        print(f"  Total Update Steps            | {conv['total_updates']:>19,d} | {gtd['total_updates']:>15,d}")
        print(f"  Corrupted Noise Ingested      | {conv['noise_ingested']:>19,d} | {gtd['noise_ingested']:>15,d}")
        print(f"  Compute Backward Pass Savings | {conv['compute_savings_pct']:>18.1f}% | {gtd['compute_savings_pct']:>14.1f}%")
        diff_acc = gtd['final_test_accuracy'] - conv['final_test_accuracy']
        noise_reduction = (1.0 - (gtd['noise_ingested'] / max(1, conv['noise_ingested']))) * 100
        print("  " + "-" * 66)
        print(f"  --> Accuracy Delta (ALG vs Baseline) : {diff_acc:+.2f}%")
        print(f"  --> Noise Ingestion Reduction        : {noise_reduction:.1f}%")
        print(f"  --> Compute Energy Conserved         : {gtd['compute_savings_pct']:.1f}%")
    elif gtd:
        print(f"  Final Test Accuracy           |                  N/A | {gtd['final_test_accuracy']:>14.2f}%")
        print(f"  Total Update Steps            |                  N/A | {gtd['total_updates']:>15,d}")
        print(f"  Noise Ingested                |                  N/A | {gtd['noise_ingested']:>15,d}")
        print(f"  Compute Savings               |                  N/A | {gtd['compute_savings_pct']:>14.1f}%")
    elif conv:
        print(f"  Final Test Accuracy           | {conv['final_test_accuracy']:>18.2f}% |            N/A")
        print(f"  Total Update Steps            | {conv['total_updates']:>19,d} |            N/A")
        print(f"  Noise Ingested                | {conv['noise_ingested']:>19,d} |            N/A")

    print("=" * 68)

    # 4. Save results to JSON
    out_path = Path(export_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to : {out_path.resolve()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continual stream training.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Baseline checkpoint.")
    parser.add_argument("--mode", choices=["conventional", "gated", "compare"], default="compare", help="Stream training mode.")
    parser.add_argument("--stream-size", type=int, default=2000, help="Stream size.")
    parser.add_argument("--batch-size", type=int, default=64, help="Stream batch size.")
    parser.add_argument("--label-noise", type=float, default=0.20, help="Label noise rate.")
    parser.add_argument("--feature-corruption", type=float, default=0.15, help="Feature corruption rate.")
    parser.add_argument("--export", default="research/data/stream_comparison.json", help="Export path for JSON.")
    args = parser.parse_args()

    run_experiment(
        mode=args.mode,
        stream_size=args.stream_size,
        batch_size=args.batch_size,
        label_noise=args.label_noise,
        feature_corruption=args.feature_corruption,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        export_path=args.export,
    )


if __name__ == "__main__":
    main()
