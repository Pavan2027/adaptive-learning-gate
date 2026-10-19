#!/usr/bin/env python3
"""
Show a human-readable summary of the ALG project configuration.

Usage::

    python scripts/show_config.py
    python scripts/show_config.py --config configs/default.yaml
    python scripts/show_config.py --config path/to/custom.yaml
    python scripts/show_config.py --json

Exit codes:
    0  — Configuration loaded and printed successfully.
    1  — File not found or validation error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pydantic import ValidationError

from alg.config import load_config


def _format_section(title: str, data: dict, indent: int = 2) -> str:
    """Format a config section as a readable block."""
    pad = " " * indent
    lines = [f"\n  [{title}]"]
    for key, val in data.items():
        lines.append(f"{pad}{key}: {val}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="show_config",
        description="Load and display the Adaptive Learning Gate configuration.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        metavar="PATH",
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output raw JSON instead of the formatted summary.",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as exc:
        print(f"[ERROR] Configuration validation failed:\n{exc}", file=sys.stderr)
        sys.exit(1)

    if args.as_json:
        print(json.dumps(cfg.model_dump(), indent=2))
        return

    dump = cfg.model_dump()
    sep = "=" * 62

    print(sep)
    print("   Adaptive Learning Gate — Configuration Summary")
    print(f"   Source: {Path(args.config).resolve()}")
    print(sep)

    for section, value in dump.items():
        if isinstance(value, dict):
            print(_format_section(section.upper(), value))
        else:
            print(f"\n  {section.upper()}: {value}")

    print(f"\n{sep}")
    print("  Gate decision rule:")
    g = cfg.gate
    print(f"    utility >= {g.learn_threshold}  → LEARN")
    print(f"    utility <= {g.ignore_threshold}  → IGNORE")
    print(f"    otherwise     → DEFER (queue capacity: {g.defer_queue_capacity})")
    print(sep)


if __name__ == "__main__":
    main()
