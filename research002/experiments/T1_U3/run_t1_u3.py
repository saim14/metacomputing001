#!/usr/bin/env python3
"""Run the preregistered T1-U3 multi-seed training-stability study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from run_t1_u2 import seed_validity, trajectory_statistics
from src.t1_experiment import extract_trajectory, train_one_model, write_environment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T1-U3 training-stability study")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/t1_u3_stability.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/t1_u3"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def training_config(study: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "experiment": study["experiment"],
        "hypotheses": study["hypotheses"],
        "seeds": [int(seed)],
        "data": study["data"],
        "model": study["model"],
        "training": study["training"],
        "probes": {
            "test_fraction": 0.4,
            "minimum_class_count": 30,
            "ridge_alpha": 1.0,
        },
        "validity": study["independent_validity"],
    }


def find_validation_at_step(history: pd.DataFrame, step: int) -> float:
    row = history[history["step"] == step]
    if len(row) != 1:
        raise ValueError(f"expected one validation row at comparison step {step}, got {len(row)}")
    return float(row.iloc[0]["validation_final_accuracy"])


def plot_learning_curves(
    history: pd.DataFrame,
    *,
    comparison_step: int,
    accuracy_gate: float,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    for seed, group in history.groupby("seed"):
        axis.plot(
            group["step"],
            group["validation_final_accuracy"],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=f"seed {seed}",
        )
    axis.axvline(
        comparison_step,
        color="#708090",
        linestyle="--",
        linewidth=1.2,
        label="T1-U2 horizon",
    )
    axis.axhline(
        accuracy_gate,
        color="#c44e52",
        linestyle="--",
        linewidth=1.2,
        label="learning gate",
    )
    axis.set(
        title="T1-U3 fixed-schedule learning stability",
        xlabel="Training update",
        ylabel="Final-step validation accuracy",
        ylim=(0.45, 1.01),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    study = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    comparison_step = int(study["training"]["comparison_horizon_step"])
    thresholds = study["independent_validity"]

    all_history: list[dict[str, Any]] = []
    all_trajectory_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for seed_value in study["seeds"]:
        seed = int(seed_value)
        print(f"STABILITY seed={seed}", flush=True)
        seed_dir = output_dir / f"seed_{seed}"
        config = training_config(study, seed)
        model, history, probe_data = train_one_model(
            config,
            seed,
            seed_dir / "checkpoints",
        )
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(seed_dir / "training_history.csv", index=False)
        all_history.extend(history)

        trajectory = extract_trajectory(
            model,
            probe_data["tokens"],
            probe_data["labels"],
            batch_size=int(study["training"]["validation_batch_size"]),
        )
        trajectory_dir = seed_dir / "independent"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trajectory_dir / f"t1_u3_trajectory_seed_{seed}.npz",
            **trajectory,
        )
        trajectory_rows = trajectory_statistics(trajectory, seed)
        all_trajectory_rows.extend(trajectory_rows)
        validity = seed_validity(trajectory_rows, thresholds)

        validation_at_comparison = find_validation_at_step(history_frame, comparison_step)
        terminal_validation = float(history_frame.iloc[-1]["validation_final_accuracy"])
        seed_rows.append(
            {
                **validity,
                "validation_accuracy_at_2400": validation_at_comparison,
                "terminal_validation_accuracy": terminal_validation,
                "validation_gain_after_2400": terminal_validation - validation_at_comparison,
                "best_validation_accuracy": float(
                    history_frame["validation_final_accuracy"].max()
                ),
                "best_validation_step": int(
                    history_frame.loc[
                        history_frame["validation_final_accuracy"].idxmax(), "step"
                    ]
                ),
            }
        )
        print(
            f"STABILITY_RESULT seed={seed} independent={validity['final_accuracy']:.4f} "
            f"valid_layers={validity['valid_future_flip_layers']} passed={validity['passed']}",
            flush=True,
        )

    history_frame = pd.DataFrame(all_history)
    trajectory_frame = pd.DataFrame(all_trajectory_rows)
    stability_frame = pd.DataFrame(seed_rows)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    trajectory_frame.to_csv(output_dir / "independent_trajectory_statistics.csv", index=False)
    stability_frame.to_csv(output_dir / "seed_stability.csv", index=False)

    passing_fraction = float(stability_frame["passed"].mean())
    required_fraction = float(thresholds["minimum_passing_seed_fraction"])
    passed = passing_fraction >= required_fraction
    status = (
        "passed_configuration_frozen_for_confirmatory_experiment"
        if passed
        else "failed_multi_seed_stability_gate"
    )

    if passed:
        frozen = {
            "source_experiment": study["experiment"]["id"],
            "model": study["model"],
            "training": study["training"],
            "data": study["data"],
            "freeze_rule": "No architecture or training changes before confirmatory probing.",
        }
        with (output_dir / "frozen_configuration.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(frozen, handle, sort_keys=False)

    plot_learning_curves(
        history_frame,
        comparison_step=comparison_step,
        accuracy_gate=float(thresholds["minimum_final_accuracy"]),
        output_path=output_dir / "t1_u3_learning_stability.png",
    )
    independent_values = stability_frame["final_accuracy"].to_numpy(dtype=float)
    gains = stability_frame["validation_gain_after_2400"].to_numpy(dtype=float)
    summary = {
        "experiment": study["experiment"],
        "status": status,
        "configuration": {
            "seeds": study["seeds"],
            "data": study["data"],
            "model": study["model"],
            "training": study["training"],
        },
        "historical_reference": study["historical_reference"],
        "sealed_evaluation": study["sealed_evaluation"],
        "probe_policy": study["probes"]["policy"],
        "validity": {
            "thresholds": thresholds,
            "passing_seed_fraction": passing_fraction,
            "required_passing_seed_fraction": required_fraction,
            "passed": bool(passed),
            "seeds": seed_rows,
        },
        "aggregate": {
            "mean_independent_accuracy": float(np.mean(independent_values)),
            "sample_std_independent_accuracy": float(np.std(independent_values, ddof=1)),
            "minimum_independent_accuracy": float(np.min(independent_values)),
            "maximum_independent_accuracy": float(np.max(independent_values)),
            "mean_validation_gain_after_2400": float(np.mean(gains)),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_environment_manifest(output_dir / "environment.json")
    return summary


if __name__ == "__main__":
    arguments = parse_args()
    result = run_study(arguments.config, arguments.output)
    print(
        f"Completed {result['experiment']['id']} with status={result['status']} "
        f"in {result['elapsed_seconds']:.1f}s; outputs: {arguments.output.resolve()}",
        flush=True,
    )
