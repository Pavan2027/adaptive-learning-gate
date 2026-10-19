#!/usr/bin/env python3
"""
Inspect and validate the simulated continual learning stream.

Usage::

    # Inspect stream with default settings:
    python scripts/inspect_stream.py

    # Inject 20% label noise and 15% feature corruption:
    python scripts/inspect_stream.py --label-noise 0.20 --feature-corruption 0.15

    # Simulate a 2-segment distribution shift and export manifest:
    python scripts/inspect_stream.py --shift-type class_shift --segments 2 --export-manifest research/data/stream_manifest.json

    # Fast small stream:
    python scripts/inspect_stream.py --size 1000 --export-manifest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alg.config import load_config
from alg.data.datasets import load_raw_dataset
from alg.data.stream import StreamSimulator
from alg.utils.seeding import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stream_inspector")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect continual learning data stream and verify perturbations."
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
        "--size",
        type=int,
        help="Stream size (number of samples).",
    )
    parser.add_argument(
        "--label-noise",
        type=float,
        help="Fraction of labels to corrupt [0.0, 1.0].",
    )
    parser.add_argument(
        "--feature-corruption",
        type=float,
        help="Fraction of feature tensors to corrupt [0.0, 1.0].",
    )
    parser.add_argument(
        "--redundancy",
        type=float,
        help="Fraction of samples to duplicate [0.0, 1.0].",
    )
    parser.add_argument(
        "--shift-type",
        choices=["none", "class_shift", "feature_shift"],
        help="Distribution shift pattern across stream segments.",
    )
    parser.add_argument(
        "--segments",
        type=int,
        help="Number of stream segments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic stream construction.",
    )
    parser.add_argument(
        "--export-manifest",
        nargs="?",
        const="results/stream_manifest.json",
        help="Export stream manifest JSON (default: results/stream_manifest.json).",
    )
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.config)
    dataset_name = args.dataset or cfg.dataset.name
    size = args.size or cfg.stream.size
    label_noise = args.label_noise if args.label_noise is not None else cfg.stream.label_noise_rate
    feature_corr = (
        args.feature_corruption
        if args.feature_corruption is not None
        else cfg.stream.feature_corruption_rate
    )
    redundancy = args.redundancy if args.redundancy is not None else cfg.stream.redundancy_rate
    shift_type = args.shift_type or cfg.stream.shift_type
    num_segments = args.segments or cfg.stream.num_segments
    seed = args.seed if args.seed is not None else cfg.seed

    set_seed(seed)

    print("=" * 64)
    print("   Adaptive Learning Gate — Continual Stream Simulator")
    print("=" * 64)
    print(f"  Dataset              : {dataset_name}")
    print(f"  Stream Size          : {size:,} samples")
    print(f"  Label Noise Rate     : {label_noise * 100:.1f}%")
    print(f"  Feature Corruption   : {feature_corr * 100:.1f}%")
    print(f"  Redundancy Rate      : {redundancy * 100:.1f}%")
    print(f"  Shift Type           : {shift_type}")
    print(f"  Stream Segments      : {num_segments}")
    print(f"  Seed                 : {seed}")
    print("=" * 64)

    # 2. Load base training dataset
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])
    base_dataset = load_raw_dataset(
        dataset_name=dataset_name,
        root=cfg.dataset.root,
        train=True,
        download=cfg.dataset.download,
        transform=transform,
    )

    # 3. Build Stream Simulator
    simulator = StreamSimulator(
        base_dataset=base_dataset,
        stream_size=size,
        label_noise_rate=label_noise,
        feature_corruption_rate=feature_corr,
        redundancy_rate=redundancy,
        shift_type=shift_type,
        num_segments=num_segments,
        seed=seed,
        num_classes=10,
    )

    stats = simulator.get_statistics()

    print("\n  [Stream Composition Summary]")
    print(f"    Total Samples      : {stats['total_samples']:,}")
    print(f"    Clean Samples      : {stats['clean_samples']:,} ({stats['clean_percentage']}%)")
    print(f"    Noisy Labels       : {stats['noisy_label_samples']:,} ({stats['noisy_label_percentage']}%)")
    print(f"    Corrupted Features : {stats['corrupted_feature_samples']:,} ({stats['corrupted_feature_percentage']}%)")
    print(f"    Redundant / Dupes  : {stats['redundant_samples']:,} ({stats['redundant_percentage']}%)")

    print("\n  [Segment Distribution]")
    for seg, count in stats["segments"].items():
        print(f"    Segment {seg}          : {count:,} samples ({count / stats['total_samples'] * 100:.1f}%)")

    # 4. Export manifest if requested
    if args.export_manifest:
        out_path = Path(args.export_manifest)
        simulator.export_manifest(out_path)
        print(f"\n  Manifest exported to : {out_path.resolve()}")

    print("=" * 64)


if __name__ == "__main__":
    main()
