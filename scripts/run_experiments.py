#!/usr/bin/env python3
"""
Orchestrate Comparative Experiments and Ablations across the defined matrix.

Supported Methods:
- update_all        : Unconditional continual baseline.
- random            : Budget-matched random selection.
- uncertainty_only  : Classical Active Learning (entropy-based).
- gated_full        : Multi-signal Adaptive Learning Gate (ours).

Supported Ablations:
- ablation_no_novelty     : Full gate without novelty signal (w_nov=0).
- ablation_no_reliability : Full gate without reliability check (w_unrel=0, no coupling).
- ablation_no_uncertainty : Full gate without uncertainty signal (w_unc=0).
- ablation_no_defer       : Hard binary gate without quarantine (tau_learn=tau_ignore).

Usage::

    # Quick smoke test (100 samples)
    python scripts/run_experiments.py --smoke

    # Run primary methods on noisy stream (seed 42)
    python scripts/run_experiments.py --condition noisy_stream --methods update_all,random,uncertainty_only,gated_full

    # Run component ablations
    python scripts/run_experiments.py --condition noisy_stream --run-ablations

    # Full matrix multi-seed benchmark
    python scripts/run_experiments.py --condition noisy_stream --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from alg.config import load_config
from alg.data.datasets import get_dataset_metadata, load_raw_dataset
from alg.data.stream import StreamSimulator
from alg.gating import (
    AdaptiveLearningGate,
    RandomGatePolicy,
    RuleBasedGatePolicy,
    UncertaintyGatePolicy,
)
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
logger = logging.getLogger("exp_runner")


def _load_experiments_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_test_loader(dataset_name: str, root: str, batch_size: int = 128) -> DataLoader:
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


def run_single_experiment(
    method_key: str,
    method_cfg: dict[str, Any],
    condition_key: str,
    condition_cfg: dict[str, Any],
    seed: int,
    stream_size: int,
    batch_size: int,
    checkpoint_path: str,
    dataset_name: str,
    dataset_root: str,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    run_id = f"{method_key}_{condition_key}_s{seed}"
    logger.info("Executing Run: %s ...", run_id)
    t0 = time.perf_counter()

    set_seed(seed)

    # 1. Prepare Dataset and Test DataLoader
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

    raw_train = load_raw_dataset(dataset_name, root=dataset_root, train=True, download=True, transform=transform)
    test_loader = _build_test_loader(dataset_name, root=dataset_root, batch_size=128)

    # 2. Build Stream Simulator for this condition and seed
    simulator = StreamSimulator(
        base_dataset=raw_train,
        stream_size=stream_size,
        label_noise_rate=condition_cfg.get("label_noise_rate", 0.0),
        feature_corruption_rate=condition_cfg.get("feature_corruption_rate", 0.0),
        redundancy_rate=condition_cfg.get("redundancy_rate", 0.0),
        seed=seed,
    )

    # Count ground-truth noisy labels in stream for metrics calculation
    total_noisy_labels = sum(1 for s in simulator.samples if s.metadata.is_noisy_label)

    # 3. Build Model from Checkpoint
    model = build_model(architecture="cnn", num_classes=10, in_channels=in_channels)
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        load_checkpoint(ckpt, model, device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)

    # 4. Configure Policy and Trainer based on method
    m_type = method_cfg.get("type", "gated")

    if m_type == "conventional":
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
        # Build prototype memory
        memory = PrototypeMemory(num_classes=10, feature_dim=256, device=device)
        with torch.no_grad():
            w_loader = DataLoader(raw_train, batch_size=128, shuffle=True)
            for b_idx, (imgs, lbls) in enumerate(w_loader):
                if b_idx >= 5:
                    break
                feats = model.forward_features(imgs.to(device))
                memory.update(feats, lbls.to(device))

        extractor = SignalExtractor(model=model, prototype_memory=memory, device=device)

        if m_type == "random":
            policy = RandomGatePolicy(budget_rate=method_cfg.get("budget_rate", 0.35), seed=seed)
            gate = AdaptiveLearningGate(policy=policy, defer_queue_capacity=0)
        elif m_type == "uncertainty_only":
            policy = UncertaintyGatePolicy(uncertainty_threshold=method_cfg.get("threshold", 0.35))
            gate = AdaptiveLearningGate(policy=policy, defer_queue_capacity=0)
        else:
            # Gated ALG or Ablations
            policy = RuleBasedGatePolicy(
                learn_threshold=method_cfg.get("learn_threshold", 0.50),
                ignore_threshold=method_cfg.get("ignore_threshold", 0.30),
                w_novelty=method_cfg.get("w_novelty", 0.5),
                w_uncertainty=method_cfg.get("w_uncertainty", 0.5),
                w_unreliability=method_cfg.get("w_unreliability", 0.8),
                w_redundancy=method_cfg.get("w_redundancy", 0.2),
                allow_no_defer=method_cfg.get("allow_no_defer", False),
                use_reliability_coupling=method_cfg.get("use_reliability_coupling", True),
            )
            gate = AdaptiveLearningGate(
                policy=policy,
                defer_queue_capacity=method_cfg.get("defer_queue_capacity", 500),
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

    elapsed = time.perf_counter() - t0

    noise_interception_pct = (
        round((1.0 - (res.noise_ingested / max(1, total_noisy_labels))) * 100, 2)
        if total_noisy_labels > 0
        else 100.0
    )

    record = {
        "run_id": run_id,
        "method": method_key,
        "method_name": method_cfg.get("name", method_key),
        "condition": condition_key,
        "condition_name": condition_cfg.get("name", condition_key),
        "seed": seed,
        "stream_size": stream_size,
        "batch_size": batch_size,
        "final_test_accuracy": round(res.final_test_accuracy * 100, 2),
        "final_test_f1": round(res.final_test_f1 * 100, 2),
        "total_updates": res.total_update_steps,
        "samples_learned": res.samples_learned,
        "samples_ignored": res.samples_ignored,
        "samples_deferred": res.samples_deferred,
        "noise_ingested": res.noise_ingested,
        "total_noisy_labels": total_noisy_labels,
        "noise_interception_pct": noise_interception_pct,
        "sample_savings_pct": res.sample_savings_pct,
        "elapsed_seconds": round(elapsed, 2),
        "trajectory": res.accuracy_trajectory,
    }

    # Save individual run file
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{run_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    logger.info(
        "Run Completed: Acc=%.2f%% | Noise Ingested=%d/%d (%.1f%% blocked) | Time=%.1fs",
        record["final_test_accuracy"],
        record["noise_ingested"],
        record["total_noisy_labels"],
        record["noise_interception_pct"],
        record["elapsed_seconds"],
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALG comparative experiment suite.")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Path to experiments.yaml.")
    parser.add_argument("--condition", default="noisy_stream", help="Stream condition from yaml.")
    parser.add_argument("--methods", help="Comma-separated methods to run.")
    parser.add_argument("--run-ablations", action="store_true", help="Run ablation configurations.")
    parser.add_argument("--seeds", help="Comma-separated random seeds (e.g. 42,43).")
    parser.add_argument("--stream-size", type=int, help="Override stream size.")
    parser.add_argument("--batch-size", type=int, default=64, help="Stream batch size.")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test (100 samples).")
    args = parser.parse_args()

    exp_cfg = _load_experiments_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = exp_cfg["meta"]["baseline_checkpoint"]
    out_dir = Path(exp_cfg["meta"]["output_dir"])

    stream_size = 100 if args.smoke else (args.stream_size or exp_cfg["meta"]["default_stream_size"])
    seeds = [42] if args.smoke else ([int(s) for s in args.seeds.split(",")] if args.seeds else [42])

    condition_cfg = exp_cfg["conditions"].get(args.condition)
    if condition_cfg is None:
        raise ValueError(f"Unknown condition '{args.condition}'. Available: {list(exp_cfg['conditions'].keys())}")

    # Build target runs dictionary
    target_methods: dict[str, Any] = {}
    if args.smoke:
        target_methods["update_all"] = exp_cfg["methods"]["update_all"]
        target_methods["gated_full"] = exp_cfg["methods"]["gated_full"]
    elif args.run_ablations:
        target_methods["gated_full"] = exp_cfg["methods"]["gated_full"]
        for k, v in exp_cfg["ablations"].items():
            target_methods[k] = v
    elif args.methods:
        for m in args.methods.split(","):
            m = m.strip()
            if m in exp_cfg["methods"]:
                target_methods[m] = exp_cfg["methods"][m]
            elif m in exp_cfg["ablations"]:
                target_methods[m] = exp_cfg["ablations"][m]
            else:
                raise ValueError(f"Unknown method '{m}'.")
    else:
        target_methods = exp_cfg["methods"]

    print("=" * 72)
    print("   Adaptive Learning Gate — Systematic Experiment Suite")
    print("=" * 72)
    print(f"  Condition            : {args.condition} ({condition_cfg.get('name')})")
    print(f"  Stream Size          : {stream_size:,} samples")
    print(f"  Seeds ({len(seeds)})        : {seeds}")
    print(f"  Methods ({len(target_methods)})      : {list(target_methods.keys())}")
    print(f"  Device               : {device}")
    print(f"  Output Directory     : {out_dir.resolve()}")
    print("=" * 72)

    all_records: list[dict[str, Any]] = []

    for seed in seeds:
        for m_key, m_cfg in target_methods.items():
            rec = run_single_experiment(
                method_key=m_key,
                method_cfg=m_cfg,
                condition_key=args.condition,
                condition_cfg=condition_cfg,
                seed=seed,
                stream_size=stream_size,
                batch_size=args.batch_size,
                checkpoint_path=ckpt_path,
                dataset_name="cifar10",
                dataset_root="data/",
                output_dir=out_dir,
                device=device,
            )
            all_records.append(rec)

    # Print Formatted Table
    print("\n" + "=" * 76)
    print("   Experiment Suite Results Summary")
    print("=" * 76)
    print(f"  {'Method':<28} | {'Test Acc':>9} | {'Noise Ingested':>15} | {'Compute Saved':>13}")
    print("  " + "-" * 72)

    for r in all_records:
        acc_str = f"{r['final_test_accuracy']:.2f}%"
        noise_str = f"{r['noise_ingested']}/{r['total_noisy_labels']} ({r['noise_interception_pct']}%)"
        sav_str = f"{r['sample_savings_pct']:.1f}%"
        print(f"  {r['method_name']:<28} | {acc_str:>9} | {noise_str:>15} | {sav_str:>13}")

    print("=" * 76)
    print(f"\nAll runs saved to {out_dir.resolve()}.\n")


if __name__ == "__main__":
    main()
