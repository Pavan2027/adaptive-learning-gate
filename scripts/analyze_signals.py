#!/usr/bin/env python3
"""
Diagnostic analysis of per-sample learning signals.

Evaluates a trained model checkpoint over a perturbed continual stream to measure:
- How signals distribute across Clean, Noisy-Label, and Feature-Corrupted samples.
- Signal correlation and diagnostic separation power.

Usage::

    python scripts/analyze_signals.py
    python scripts/analyze_signals.py --checkpoint checkpoints/best_model.pt --stream-size 2000
    python scripts/analyze_signals.py --export research/data/signal_analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torchvision.transforms as transforms

from alg.config import load_config
from alg.data.datasets import load_raw_dataset
from alg.data.stream import StreamSimulator
from alg.models import build_model
from alg.signals import PrototypeMemory, SignalExtractor
from alg.training.baseline import load_checkpoint
from alg.utils.seeding import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("signal_analyzer")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze distribution of learning signals across stream conditions."
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
        help="Number of stream samples to analyze.",
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
        "--export",
        default="research/data/signal_analysis.json",
        help="Destination path for signal analysis JSON.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    print("=" * 64)
    print("   Adaptive Learning Gate — Learning Signal Diagnostic Analysis")
    print("=" * 64)
    print(f"  Model Checkpoint     : {args.checkpoint}")
    print(f"  Device               : {device}")
    print(f"  Stream Size          : {args.stream_size:,} samples")
    print(f"  Label Noise Rate     : {args.label_noise * 100:.1f}%")
    print(f"  Feature Corruption   : {args.feature_corruption * 100:.1f}%")
    print("=" * 64)

    # 1. Load trained model
    in_channels = 3 if cfg.dataset.name == "cifar10" else 1
    model = build_model(
        architecture=cfg.model.architecture,
        num_classes=10,
        in_channels=in_channels,
    )

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model, device=device)
        logger.info("Loaded trained model from '%s'.", ckpt_path)
    else:
        logger.warning("Checkpoint '%s' not found. Using freshly initialized model.", ckpt_path)

    model.to(device)
    model.eval()

    # 2. Build prototype memory from a small clean warmup subset
    memory = PrototypeMemory(num_classes=10, feature_dim=256, device=device)
    raw_dataset = load_raw_dataset(
        cfg.dataset.name, root=cfg.dataset.root, train=True, download=cfg.dataset.download,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) if in_channels == 3 else transforms.Normalize((0.2860,), (0.3530,))])
    )

    with torch.no_grad():
        warmup_loader = torch.utils.data.DataLoader(raw_dataset, batch_size=128, shuffle=True)
        for b_idx, (imgs, lbls) in enumerate(warmup_loader):
            if b_idx >= 5:  # ~640 samples to populate prototypes
                break
            imgs, lbls = imgs.to(device), lbls.to(device)
            feats = model.forward_features(imgs)
            memory.update(feats, lbls)

    # 3. Build perturbed stream
    simulator = StreamSimulator(
        base_dataset=raw_dataset,
        stream_size=args.stream_size,
        label_noise_rate=args.label_noise,
        feature_corruption_rate=args.feature_corruption,
        seed=cfg.seed,
    )

    extractor = SignalExtractor(model=model, prototype_memory=memory, device=device)

    # 4. Extract signals across stream
    records = []
    stream_loader = simulator.get_dataloader(batch_size=128)

    sample_offset = 0
    for images, observed_labels in stream_loader:
        scores = extractor.extract_batch(images, observed_labels)
        for i, sc in enumerate(scores):
            raw_sample = simulator.samples[sample_offset + i]
            records.append({
                "uncertainty": sc.uncertainty,
                "novelty": sc.novelty,
                "reliability": sc.reliability,
                "is_noisy_label": raw_sample.metadata.is_noisy_label,
                "is_corrupted_feature": raw_sample.metadata.is_corrupted_feature,
                "is_clean": (not raw_sample.metadata.is_noisy_label and not raw_sample.metadata.is_corrupted_feature),
            })
        sample_offset += len(scores)

    # 5. Compute grouped signal distributions
    def group_stats(data: list[dict], key: str, filter_fn) -> tuple[float, float]:
        subset = [d[key] for d in data if filter_fn(d)]
        if not subset:
            return 0.0, 0.0
        return float(np.mean(subset)), float(np.std(subset))

    clean_u_m, clean_u_s = group_stats(records, "uncertainty", lambda d: d["is_clean"])
    noisy_u_m, noisy_u_s = group_stats(records, "uncertainty", lambda d: d["is_noisy_label"])
    corr_u_m, corr_u_s = group_stats(records, "uncertainty", lambda d: d["is_corrupted_feature"])

    clean_r_m, clean_r_s = group_stats(records, "reliability", lambda d: d["is_clean"])
    noisy_r_m, noisy_r_s = group_stats(records, "reliability", lambda d: d["is_noisy_label"])
    corr_r_m, corr_r_s = group_stats(records, "reliability", lambda d: d["is_corrupted_feature"])

    clean_n_m, clean_n_s = group_stats(records, "novelty", lambda d: d["is_clean"])
    corr_n_m, corr_n_s = group_stats(records, "novelty", lambda d: d["is_corrupted_feature"])

    print("\n  [Signal Distribution across Ground-Truth Conditions (Mean +/- Std)]")
    print(f"  Condition        | Uncertainty (Entropy) | Reliability Score    | Novelty Distance")
    print("  " + "-" * 70)
    print(f"  Clean            | {clean_u_m:.4f} +/- {clean_u_s:.4f}    | {clean_r_m:.4f} +/- {clean_r_s:.4f}   | {clean_n_m:.4f} +/- {clean_n_s:.4f}")
    print(f"  Noisy Label      | {noisy_u_m:.4f} +/- {noisy_u_s:.4f}    | {noisy_r_m:.4f} +/- {noisy_r_s:.4f}   | -")
    print(f"  Corrupted Feature| {corr_u_m:.4f} +/- {corr_u_s:.4f}    | {corr_r_m:.4f} +/- {corr_r_s:.4f}   | {corr_n_m:.4f} +/- {corr_n_s:.4f}")
    print("  " + "-" * 70)

    # Key diagnostic check
    rel_gap = clean_r_m - noisy_r_m
    print(f"\n  [Diagnostic Validation]")
    print(f"    Clean vs Noisy-Label Reliability Gap: +{rel_gap:.4f} (High separation: {rel_gap > 0.2})")

    # 6. Export report JSON
    analysis_results = {
        "num_samples_analyzed": len(records),
        "metrics": {
            "clean": {
                "uncertainty": {"mean": round(clean_u_m, 4), "std": round(clean_u_s, 4)},
                "reliability": {"mean": round(clean_r_m, 4), "std": round(clean_r_s, 4)},
                "novelty": {"mean": round(clean_n_m, 4), "std": round(clean_n_s, 4)},
            },
            "noisy_label": {
                "uncertainty": {"mean": round(noisy_u_m, 4), "std": round(noisy_u_s, 4)},
                "reliability": {"mean": round(noisy_r_m, 4), "std": round(noisy_r_s, 4)},
            },
            "corrupted_feature": {
                "uncertainty": {"mean": round(corr_u_m, 4), "std": round(corr_u_s, 4)},
                "reliability": {"mean": round(corr_r_m, 4), "std": round(corr_r_s, 4)},
                "novelty": {"mean": round(corr_n_m, 4), "std": round(corr_n_s, 4)},
            },
        },
        "reliability_separation_gap": round(rel_gap, 4),
    }

    out_file = Path(args.export)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)

    print(f"  Analysis saved to   : {out_file.resolve()}")
    print("=" * 64)


if __name__ == "__main__":
    main()
