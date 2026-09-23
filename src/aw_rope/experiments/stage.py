"""Run all work belonging to one serial experiment stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from .grid import run_stage
from .theory import run as run_theory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.stage == "stage1":
        theory_path = args.output_root / "stage1" / "theory.json"
        if not theory_path.exists():
            run_theory(theory_path)
    return run_stage(
        args.stage,
        data_root=args.data_root,
        output_root=args.output_root,
        workers=args.workers,
        mode="all",
    )


if __name__ == "__main__":
    raise SystemExit(main())
