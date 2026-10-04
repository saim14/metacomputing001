#!/usr/bin/env python3
"""CLI entrypoint for T1."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.t1_experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T1 Transformer trajectory sufficiency")
    parser.add_argument("--config", type=Path, default=Path("configs/t1_baseline.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/t1_baseline"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary = run_experiment(arguments.config, arguments.output)
    print(
        f"Completed {summary['experiment']['id']} in {summary['elapsed_seconds']:.1f}s; "
        f"outputs: {arguments.output.resolve()}",
        flush=True,
    )

