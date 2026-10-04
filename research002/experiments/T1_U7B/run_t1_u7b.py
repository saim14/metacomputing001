#!/usr/bin/env python3
"""Run T1-U7B: fresh-stream extension of the bounded two-hop gate.

The complete T1-U7 4,000-update prefix is replayed exactly, preserving Adam
state. Four new final-only two-hop streams then add 2,000 fixed updates. Fresh
model seeds remain conditional on an all-development-seed pass. No trajectory
probe can be fit by this runner.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from run_t1_u7 import (
    frame_records,
    load_config,
    run_seed_set,
    validate_protocol as validate_u7_protocol,
)
from src.t1_experiment import write_environment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T1-U7B fresh-stream extension")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/t1_u7b_fresh_stream_extension.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/t1_u7b"))
    return parser.parse_args()


def validate_protocol(
    study: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    validate_u7_protocol(study)
    preserved_fields = [
        "development_seeds",
        "fresh_confirmation_seeds",
        "data",
        "model",
        "development_gate",
        "fresh_confirmation",
    ]
    mismatches = [field for field in preserved_fields if study[field] != reference[field]]

    study_training = copy.deepcopy(study["training"])
    reference_training = copy.deepcopy(reference["training"])
    study_phases = study_training.pop("curriculum")
    reference_phases = reference_training.pop("curriculum")
    if study_training != reference_training:
        mismatches.append("training_noncurriculum_fields")
    if study_phases[: len(reference_phases)] != reference_phases:
        mismatches.append("training_t1_u7_prefix")
    if mismatches:
        raise ValueError(f"T1-U7 fixed-field mismatch: {mismatches}")

    expected_base = int(study["closed_reference"]["expected_base_updates"])
    actual_base = int(sum(int(phase["steps"]) for phase in reference_phases))
    if actual_base != expected_base:
        raise ValueError("closed-reference update count mismatch")

    extension_phases = study_phases[len(reference_phases) :]
    extension = study["extension"]
    expected_streams = [int(value) for value in extension["stream_ids"]]
    actual_streams = [int(phase["data_stream_id"]) for phase in extension_phases]
    if actual_streams != expected_streams:
        raise ValueError("extension stream order differs from preregistration")
    if set(actual_streams) & {
        int(phase["data_stream_id"]) for phase in reference_phases
    }:
        raise ValueError("extension streams must be fresh relative to T1-U7")
    if any(int(phase["steps"]) != int(extension["updates_per_stream"]) for phase in extension_phases):
        raise ValueError("extension phase length mismatch")
    if any(float(phase["auxiliary_loss_weight"]) != 0.0 for phase in extension_phases):
        raise ValueError("extension must be final-only")
    if any(int(phase["data_overrides"]["chain_hops"]) != 2 for phase in extension_phases):
        raise ValueError("extension must remain two-hop")
    actual_extension = int(sum(int(phase["steps"]) for phase in extension_phases))
    if actual_extension != int(extension["total_updates"]):
        raise ValueError("total extension budget mismatch")

    return {
        "fixed_fields_verified": preserved_fields + [
            "training_noncurriculum_fields",
            "training_t1_u7_prefix",
        ],
        "base_updates": actual_base,
        "extension_updates": actual_extension,
        "extension_stream_ids": actual_streams,
        "total_updates": actual_base + actual_extension,
        "passed": True,
    }


def load_reference_results(study: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(str(study["closed_reference"]["output"]))
    gate_path = root / "development_seed_gate.csv"
    causal_path = root / "development_causal_metrics.csv"
    if not gate_path.exists() or not causal_path.exists():
        raise FileNotFoundError("closed T1-U7 development results are unavailable")
    return pd.read_csv(gate_path), pd.read_csv(causal_path)


def paired_development_effects(
    baseline_gate: pd.DataFrame,
    extended_gate: pd.DataFrame,
    baseline_causal: pd.DataFrame,
    extended_causal: pd.DataFrame,
) -> pd.DataFrame:
    baseline = baseline_gate.merge(
        baseline_causal[[
            "seed",
            "causal_clean_accuracy",
            "counterfactual_accuracy",
            "counterfactual_prediction_flip_consistency",
        ]],
        on="seed",
        how="left",
    )
    extended = extended_gate.merge(
        extended_causal[[
            "seed",
            "causal_clean_accuracy",
            "counterfactual_accuracy",
            "counterfactual_prediction_flip_consistency",
        ]],
        on="seed",
        how="left",
    )
    columns = [
        "final_accuracy",
        "counterfactual_pair_accuracy",
        "first_edge_corrupted_accuracy",
        "second_edge_corrupted_accuracy",
        "valid_value_step_count",
        "causal_clean_accuracy",
        "counterfactual_accuracy",
        "counterfactual_prediction_flip_consistency",
    ]
    merged = baseline[["seed", "passed", *columns]].merge(
        extended[["seed", "passed", *columns]],
        on="seed",
        suffixes=("_t1_u7", "_t1_u7b"),
        validate="one_to_one",
    )
    for column in columns:
        merged[f"delta_{column}"] = (
            merged[f"{column}_t1_u7b"] - merged[f"{column}_t1_u7"]
        )
    return merged


def plot_results(
    training: pd.DataFrame,
    trajectory: pd.DataFrame,
    causal: pd.DataFrame,
    effects: pd.DataFrame,
    study: dict[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), constrained_layout=True)
    base_updates = int(study["closed_reference"]["expected_base_updates"])
    gate = study["development_gate"]
    styles = {"development": "-", "fresh_confirmation": "--"}

    for (phase, seed), group in training.groupby(["evaluation_phase", "seed"]):
        axes[0, 0].plot(
            group["step"],
            group["validation_final_accuracy"],
            linestyle=styles.get(str(phase), "-"),
            label=f"{phase}: {seed}",
        )
    axes[0, 0].axvline(base_updates, color="#777777", linestyle="--", label="T1-U7 boundary")
    axes[0, 0].axhline(float(gate["minimum_final_accuracy"]), color="#c44e52", linestyle=":")
    axes[0, 0].set(
        title="Learning with fresh-stream extension",
        xlabel="Training update",
        ylabel="Validation accuracy",
        ylim=(0.45, 1.01),
    )
    axes[0, 0].legend(frameon=False, fontsize=7)

    x = np.arange(len(effects))
    width = 0.34
    axes[0, 1].bar(
        x - width / 2,
        effects["final_accuracy_t1_u7"],
        width,
        label="T1-U7",
        color="#9ecae1",
    )
    axes[0, 1].bar(
        x + width / 2,
        effects["final_accuracy_t1_u7b"],
        width,
        label="T1-U7B",
        color="#3182bd",
    )
    axes[0, 1].axhline(float(gate["minimum_final_accuracy"]), color="#c44e52", linestyle=":")
    axes[0, 1].set(
        title="Paired clean-accuracy repair",
        ylabel="Fixed development accuracy",
        xticks=x,
        xticklabels=[str(int(seed)) for seed in effects["seed"]],
        ylim=(0.85, 1.005),
    )
    axes[0, 1].legend(frameon=False)

    for (phase, seed), group in trajectory.groupby(["evaluation_phase", "seed"]):
        axes[1, 0].plot(
            group["step"],
            group["mean_remaining_nll_reduction"],
            marker="o",
            linestyle=styles.get(str(phase), "-"),
            label=f"{phase}: {seed}",
        )
    axes[1, 0].axhline(
        float(gate["minimum_mean_remaining_nll_reduction"]),
        color="#c44e52",
        linestyle=":",
    )
    axes[1, 0].set(
        title="Remaining value of computation",
        xlabel="Current refinement step",
        ylabel=r"Mean $\mathcal{L}_l-\mathcal{L}_L$ (nats)",
    )

    metrics = [
        "causal_clean_accuracy",
        "counterfactual_accuracy",
        "first_edge_corrupted_accuracy",
        "second_edge_corrupted_accuracy",
    ]
    labels = ["Clean", "Counterfactual", "Break first", "Break second"]
    development = causal[causal["evaluation_phase"] == "development"]
    means = [float(development[metric].mean()) for metric in metrics]
    axes[1, 1].bar(
        np.arange(len(metrics)),
        means,
        color=["#4c78a8", "#59a14f", "#bab0ab", "#9c755f"],
    )
    for index, metric in enumerate(metrics):
        values = development[metric].to_numpy(dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0.0]
        axes[1, 1].scatter(index + np.asarray(jitter), values, color="#222222", s=18)
    axes[1, 1].axhspan(
        float(gate["corruption_accuracy_minimum"]),
        float(gate["corruption_accuracy_maximum"]),
        color="#dddddd",
        alpha=0.5,
    )
    axes[1, 1].set(
        title="Causal controls after extension",
        ylabel="Final accuracy",
        xticks=np.arange(len(metrics)),
        xticklabels=labels,
        ylim=(0.0, 1.01),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    fig.suptitle("T1-U7B fresh-stream training-stability repair", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    study = load_config(config_path)
    reference_config_path = Path(str(study["closed_reference"]["config"]))
    reference = load_config(reference_config_path)
    reference_verification = validate_protocol(study, reference)
    baseline_gate, baseline_causal = load_reference_results(study)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / "preregistered_protocol.yaml")
    (output_dir / "closed_reference_verification.json").write_text(
        json.dumps(reference_verification, indent=2), encoding="utf-8"
    )
    started = time.perf_counter()

    development_seeds = [int(seed) for seed in study["development_seeds"]]
    dev_history, dev_trajectory, dev_causal, dev_gate = run_seed_set(
        study,
        seeds=development_seeds,
        phase="development",
        sample_offset=int(study["data"]["development_sample_seed_offset"]),
        causal_offset=int(study["data"]["development_causal_seed_offset"]),
        output_dir=output_dir,
    )
    effects = paired_development_effects(
        baseline_gate, dev_gate, baseline_causal, dev_causal
    )
    dev_history.to_csv(output_dir / "development_training_history.csv", index=False)
    dev_trajectory.to_csv(
        output_dir / "development_trajectory_value_statistics.csv", index=False
    )
    dev_causal.to_csv(output_dir / "development_causal_metrics.csv", index=False)
    dev_gate.to_csv(output_dir / "development_seed_gate.csv", index=False)
    effects.to_csv(output_dir / "paired_development_effects.csv", index=False)
    development_passing_fraction = float(dev_gate["passed"].mean())
    development_passed = development_passing_fraction >= float(
        study["development_gate"]["minimum_passing_seed_fraction"]
    )

    fresh_opened = False
    fresh_passed = False
    fresh_history = pd.DataFrame()
    fresh_trajectory = pd.DataFrame()
    fresh_causal = pd.DataFrame()
    fresh_gate = pd.DataFrame()
    if development_passed:
        fresh_opened = True
        fresh_seeds = [int(seed) for seed in study["fresh_confirmation_seeds"]]
        fresh_history, fresh_trajectory, fresh_causal, fresh_gate = run_seed_set(
            study,
            seeds=fresh_seeds,
            phase="fresh_confirmation",
            sample_offset=int(study["data"]["fresh_sample_seed_offset"]),
            causal_offset=int(study["data"]["fresh_causal_seed_offset"]),
            output_dir=output_dir,
        )
        fresh_history.to_csv(output_dir / "fresh_training_history.csv", index=False)
        fresh_trajectory.to_csv(
            output_dir / "fresh_trajectory_value_statistics.csv", index=False
        )
        fresh_causal.to_csv(output_dir / "fresh_causal_metrics.csv", index=False)
        fresh_gate.to_csv(output_dir / "fresh_seed_gate.csv", index=False)
        fresh_passed = bool(fresh_gate["passed"].all())

    all_history = pd.concat([dev_history, fresh_history], ignore_index=True)
    all_trajectory = pd.concat([dev_trajectory, fresh_trajectory], ignore_index=True)
    all_causal = pd.concat([dev_causal, fresh_causal], ignore_index=True)
    plot_results(
        all_history,
        all_trajectory,
        all_causal,
        effects,
        study,
        output_dir / "t1_u7b_fresh_stream_extension.png",
    )

    smoke = str(study["experiment"]["id"]).endswith("-smoke")
    if smoke:
        status = "smoke_execution_path_complete"
    elif not development_passed:
        status = "failed_development_gate_fresh_seeds_unopened"
    elif not fresh_passed:
        status = "passed_development_failed_fresh_confirmation"
    else:
        status = "passed_development_and_fresh_confirmation_ready_for_t1_u8"
    decision = {
        "development_passed": bool(development_passed),
        "development_passing_fraction": development_passing_fraction,
        "fresh_confirmation_opened": bool(fresh_opened),
        "fresh_confirmation_passed": bool(fresh_passed),
        "trajectory_probes_fitted": False,
        "advance_to_t1_u8": bool(development_passed and fresh_passed and not smoke),
    }
    (output_dir / "advancement_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment": study["experiment"],
        "status": status,
        "hypotheses": study["hypotheses"],
        "scientific_boundary": study["scientific_boundary"],
        "closed_reference_verification": reference_verification,
        "extension": study["extension"],
        "configuration": {
            "development_seeds": development_seeds,
            "fresh_confirmation_seeds": study["fresh_confirmation_seeds"],
            "data": study["data"],
            "model": study["model"],
            "training": study["training"],
        },
        "gate_thresholds": study["development_gate"],
        "paired_development_effects": frame_records(effects),
        "development": {
            "passed": bool(development_passed),
            "passing_fraction": development_passing_fraction,
            "seed_gate": frame_records(dev_gate),
            "causal_metrics": frame_records(dev_causal),
            "trajectory_value_statistics": frame_records(dev_trajectory),
        },
        "fresh_confirmation": {
            "opened": bool(fresh_opened),
            "passed": bool(fresh_passed),
            "seed_gate": frame_records(fresh_gate),
            "causal_metrics": frame_records(fresh_causal),
            "trajectory_value_statistics": frame_records(fresh_trajectory),
        },
        "advancement_decision": decision,
        "probe_policy": study["probes"]["policy"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
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
