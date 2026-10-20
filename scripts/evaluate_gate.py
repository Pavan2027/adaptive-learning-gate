#!/usr/bin/env python3
"""
Evaluate Adaptive Learning Gate actions across synthetic stream conditions.

Measures:
- Selective action breakdown (LEARN / IGNORE / DEFER).
- Noise filtering efficacy: Fraction of noisy labels successfully IGNORED.
- Retention of clean samples: Fraction of clean samples successfully LEARNED.
- Defer queue behavior and overflow tracking.

Usage::

    python scripts/evaluate_gate.py
    python scripts/evaluate_gate.py --stream-size 2000 --label-noise 0.20
    python scripts/evaluate_gate.py --export research/data/gate_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torchvision.transforms as transforms

from alg.config import load_config
from alg.data.datasets import load_raw_dataset
from alg.data.stream import StreamSimulator
from alg.gating import AdaptiveLearningGate, GateAction
from alg.models import build_model
from alg.signals import PrototypeMemory, SignalExtractor
from alg.training.baseline import load_checkpoint
from alg.utils.seeding import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gate_evaluator")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Adaptive Learning Gate decision breakdown on perturbed streams."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--stream-size",
        type=int,
        default=2000,
        help="Number of stream samples to evaluate.",
    )
    parser.add_argument(
        "--label-noise",
        type=float,
        default=0.20,
        help="Simulated label noise rate.",
    )
    parser.add_argument(
        "--feature-corruption",
        type=float,
        default=0.15,
        help="Simulated feature corruption rate.",
    )
    parser.add_argument(
        "--learn-thresh",
        type=float,
        help="Override learn threshold.",
    )
    parser.add_argument(
        "--ignore-thresh",
        type=float,
        help="Override ignore threshold.",
    )
    parser.add_argument(
        "--export",
        default="research/data/gate_evaluation.json",
        help="Destination path for gate evaluation JSON.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    learn_thresh = args.learn_thresh or cfg.gate.learn_threshold
    ignore_thresh = args.ignore_thresh or cfg.gate.ignore_threshold

    print("=" * 64)
    print("   Adaptive Learning Gate — Decision Policy Evaluation")
    print("=" * 64)
    print(f"  Model Checkpoint     : {args.checkpoint}")
    print(f"  Stream Size          : {args.stream_size:,} samples")
    print(f"  Learn Threshold      : {learn_thresh}")
    print(f"  Ignore Threshold     : {ignore_thresh}")
    print(f"  Queue Capacity       : {cfg.gate.defer_queue_capacity}")
    print(f"  Simulated Noise      : {args.label_noise * 100:.1f}% Label, {args.feature_corruption * 100:.1f}% Feat")
    print("=" * 64)

    # 1. Load model and prototype memory
    in_channels = 3 if cfg.dataset.name == "cifar10" else 1
    model = build_model(architecture=cfg.model.architecture, num_classes=10, in_channels=in_channels)
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        load_checkpoint(ckpt, model, device=device)
    model.to(device)
    model.eval()

    memory = PrototypeMemory(num_classes=10, feature_dim=256, device=device)
    raw_dataset = load_raw_dataset(
        cfg.dataset.name, root=cfg.dataset.root, train=True, download=cfg.dataset.download,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) if in_channels == 3 else transforms.Normalize((0.2860,), (0.3530,))])
    )

    # Warmup prototype memory
    with torch.no_grad():
        w_loader = torch.utils.data.DataLoader(raw_dataset, batch_size=128, shuffle=True)
        for b_idx, (imgs, lbls) in enumerate(w_loader):
            if b_idx >= 5:
                break
            feats = model.forward_features(imgs.to(device))
            memory.update(feats, lbls.to(device))

    # 2. Build stream and gate
    simulator = StreamSimulator(
        base_dataset=raw_dataset,
        stream_size=args.stream_size,
        label_noise_rate=args.label_noise,
        feature_corruption_rate=args.feature_corruption,
        seed=cfg.seed,
    )

    extractor = SignalExtractor(model=model, prototype_memory=memory, device=device)
    gate = AdaptiveLearningGate(
        learn_threshold=learn_thresh,
        ignore_threshold=ignore_thresh,
        defer_queue_capacity=cfg.gate.defer_queue_capacity,
    )

    # 3. Stream processing simulation
    decisions = []
    stream_loader = simulator.get_dataloader(batch_size=128)

    sample_offset = 0
    for images, observed_labels in stream_loader:
        batch_scores = extractor.extract_batch(images, observed_labels)
        for i, sc in enumerate(batch_scores):
            raw_sample = simulator.samples[sample_offset + i]
            decision = gate.evaluate_sample(signals=sc, sample=raw_sample)
            decisions.append({
                "decision": decision,
                "metadata": raw_sample.metadata,
            })
        sample_offset += len(batch_scores)

    # 4. Cross-tabulate Gate Actions vs Ground Truth Conditions
    breakdown = {
        "clean": {"LEARN": 0, "IGNORE": 0, "DEFER": 0, "total": 0},
        "noisy_label": {"LEARN": 0, "IGNORE": 0, "DEFER": 0, "total": 0},
        "corrupted_feat": {"LEARN": 0, "IGNORE": 0, "DEFER": 0, "total": 0},
    }

    for item in decisions:
        act = item["decision"].action.value
        meta = item["metadata"]

        if meta.is_noisy_label:
            breakdown["noisy_label"][act] += 1
            breakdown["noisy_label"]["total"] += 1
        elif meta.is_corrupted_feature:
            breakdown["corrupted_feat"][act] += 1
            breakdown["corrupted_feat"]["total"] += 1
        else:
            breakdown["clean"][act] += 1
            breakdown["clean"]["total"] += 1

    summary_stats = gate.get_summary_statistics()

    print("\n  [Gate Action Summary]")
    print(f"    Total Evaluated : {summary_stats['total_decisions']:,}")
    print(f"    LEARN Action    : {summary_stats['actions']['learn']:,} ({summary_stats['actions']['learn_percentage']}%)")
    print(f"    IGNORE Action   : {summary_stats['actions']['ignore']:,} ({summary_stats['actions']['ignore_percentage']}%)")
    print(f"    DEFER Action    : {summary_stats['actions']['defer']:,} ({summary_stats['actions']['defer_percentage']}%)")
    print(f"    Queue Overflows : {summary_stats['queue']['overflow_drops']:,}")

    print("\n  [Action Breakdown by Ground Truth Condition]")
    print(f"  Condition        | LEARN (Accepted) | IGNORE (Filtered) | DEFER (Quarantined)")
    print("  " + "-" * 70)

    for cond, row in [("Clean", breakdown["clean"]), ("Noisy Label", breakdown["noisy_label"]), ("Corrupted Feature", breakdown["corrupted_feat"])]:
        tot = max(1, row["total"])
        l_pct = row["LEARN"] / tot * 100
        i_pct = row["IGNORE"] / tot * 100
        d_pct = row["DEFER"] / tot * 100
        print(f"  {cond:<16} | {row['LEARN']:>4} ({l_pct:5.1f}%)    | {row['IGNORE']:>4} ({i_pct:5.1f}%)     | {row['DEFER']:>4} ({d_pct:5.1f}%)")

    print("  " + "-" * 70)

    # Key paper metrics
    noise_filter_rate = (
        (breakdown["noisy_label"]["IGNORE"] + breakdown["noisy_label"]["DEFER"])
        / max(1, breakdown["noisy_label"]["total"])
        * 100
    )
    clean_admit_rate = breakdown["clean"]["LEARN"] / max(1, breakdown["clean"]["total"]) * 100

    print(f"\n  [Efficacy Summary]")
    print(f"    Noisy Label Interception Rate (IGNORE+DEFER): {noise_filter_rate:.2f}%")
    print(f"    Clean Sample Admission Rate (LEARN)         : {clean_admit_rate:.2f}%")

    # 5. Export JSON
    eval_record = {
        "stream_size": args.stream_size,
        "learn_threshold": learn_thresh,
        "ignore_threshold": ignore_thresh,
        "summary": summary_stats,
        "condition_breakdown": breakdown,
        "noisy_label_interception_rate": round(noise_filter_rate, 2),
        "clean_admission_rate": round(clean_admit_rate, 2),
    }

    out_file = Path(args.export)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(eval_record, f, indent=2)

    print(f"\n  Evaluation saved to  : {out_file.resolve()}")
    print("=" * 64)


if __name__ == "__main__":
    main()
