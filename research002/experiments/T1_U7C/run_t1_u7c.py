#!/usr/bin/env python3
"""Run T1-U7C: intermediate-supervision repair of the T1-U7B extension.

The complete 6,000-update T1-U7B schedule is replayed from initialization.
Only the auxiliary-loss weight in the final four fresh-stream phases changes,
from 0.0 to 0.005. All previously evaluated model seeds are development seeds;
new confirmation seeds remain conditional on an all-development-seed pass.
This runner has no trajectory-probe fitting path.
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

from run_t1_u7 import (
    frame_records,
    load_config,
    run_seed_set,
    validate_protocol as validate_u7_protocol,
)
from src.t1_experiment import write_environment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run T1-U7C intermediate-supervision repair"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/t1_u7c_training_signal.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/t1_u7c"))
    return parser.parse_args()


def _phase_without_auxiliary_weight(phase: dict[str, Any]) -> dict[str, Any]:
    stripped = copy.deepcopy(phase)
    stripped.pop("auxiliary_loss_weight")
    return stripped


def validate_protocol(
    study: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Prove that extension supervision is the sole T1-U7B protocol change."""
    validate_u7_protocol(study)

    expected_exposed = [
        *[int(seed) for seed in reference["development_seeds"]],
        *[int(seed) for seed in reference["fresh_confirmation_seeds"]],
    ]
    development_seeds = [int(seed) for seed in study["development_seeds"]]
    fresh_seeds = [int(seed) for seed in study["fresh_confirmation_seeds"]]
    if development_seeds != expected_exposed:
        raise ValueError(
            "T1-U7C development seeds must equal all T1-U7B exposed seeds"
        )
    if len(set(development_seeds)) != len(development_seeds):
        raise ValueError("development seeds must be unique")
    if not fresh_seeds or len(set(fresh_seeds)) != len(fresh_seeds):
        raise ValueError("fresh confirmation seeds must be nonempty and unique")
    if set(development_seeds) & set(fresh_seeds):
        raise ValueError("fresh confirmation seeds overlap exposed seeds")
    if [int(seed) for seed in study["scientific_boundary"]["development_seeds"]] != development_seeds:
        raise ValueError("scientific boundary does not identify all exposed seeds")

    preserved_fields = ["data", "model", "development_gate", "fresh_confirmation"]
    mismatches = [field for field in preserved_fields if study[field] != reference[field]]

    study_training = copy.deepcopy(study["training"])
    reference_training = copy.deepcopy(reference["training"])
    study_phases = study_training.pop("curriculum")
    reference_phases = reference_training.pop("curriculum")
    if study_training != reference_training:
        mismatches.append("training_noncurriculum_fields")
    if len(study_phases) != len(reference_phases):
        mismatches.append("training_phase_count")
    if mismatches:
        raise ValueError(f"T1-U7B fixed-field mismatch: {mismatches}")

    expected_prefix = int(study["closed_reference"]["expected_prefix_updates"])
    expected_total = int(study["closed_reference"]["expected_total_updates"])
    if sum(int(phase["steps"]) for phase in reference_phases) != expected_total:
        raise ValueError("closed-reference total update count mismatch")
    if sum(int(phase["steps"]) for phase in study_phases) != expected_total:
        raise ValueError("T1-U7C total update count mismatch")

    cumulative_updates = 0
    prefix_phase_count = 0
    target_weight = float(
        study["training_signal_intervention"]["target_auxiliary_loss_weight"]
    )
    for index, (candidate, baseline) in enumerate(zip(study_phases, reference_phases)):
        if _phase_without_auxiliary_weight(candidate) != _phase_without_auxiliary_weight(baseline):
            raise ValueError(f"non-auxiliary field differs in training phase {index}")
        phase_end = cumulative_updates + int(baseline["steps"])
        if phase_end <= expected_prefix:
            prefix_phase_count += 1
            if candidate != baseline:
                raise ValueError(f"T1-U7B prefix differs in training phase {index}")
        else:
            if cumulative_updates < expected_prefix:
                raise ValueError("prefix boundary falls inside a training phase")
            if float(baseline["auxiliary_loss_weight"]) != 0.0:
                raise ValueError("T1-U7B extension reference must be final-only")
            if float(candidate["auxiliary_loss_weight"]) != target_weight:
                raise ValueError("T1-U7C extension supervision differs from preregistration")
        cumulative_updates = phase_end
    if cumulative_updates != expected_total:
        raise ValueError("training schedule total mismatch")
    if sum(int(phase["steps"]) for phase in reference_phases[:prefix_phase_count]) != expected_prefix:
        raise ValueError("prefix phase count does not resolve the expected boundary")

    intervention_phases = study_phases[prefix_phase_count:]
    intervention = study["training_signal_intervention"]
    actual_streams = [int(phase["data_stream_id"]) for phase in intervention_phases]
    expected_streams = [int(value) for value in intervention["stream_ids"]]
    if actual_streams != expected_streams:
        raise ValueError("intervention stream order differs from preregistration")
    if any(
        int(phase["steps"]) != int(intervention["updates_per_stream"])
        for phase in intervention_phases
    ):
        raise ValueError("intervention phase length mismatch")
    intervention_updates = sum(int(phase["steps"]) for phase in intervention_phases)
    if intervention_updates != int(intervention["total_updates"]):
        raise ValueError("intervention update budget mismatch")

    return {
        "fixed_fields_verified": preserved_fields
        + [
            "training_noncurriculum_fields",
            "training_phase_count",
            "training_phase_names",
            "training_phase_steps",
            "training_stream_ids",
            "training_data_overrides",
            "training_learning_rates",
            "training_t1_u7b_prefix",
        ],
        "exposed_development_seeds": development_seeds,
        "new_fresh_confirmation_seeds": fresh_seeds,
        "prefix_updates": expected_prefix,
        "intervention_updates": intervention_updates,
        "intervention_stream_ids": actual_streams,
        "reference_extension_auxiliary_loss_weight": 0.0,
        "intervention_auxiliary_loss_weight": target_weight,
        "total_updates": expected_total,
        "changed_fields": [str(intervention["changed_field"])],
        "passed": True,
    }


def load_reference_results(
    study: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(str(study["closed_reference"]["output"]))
    required = [
        root / "development_seed_gate.csv",
        root / "development_causal_metrics.csv",
        root / "fresh_seed_gate.csv",
        root / "fresh_causal_metrics.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"closed T1-U7B results are unavailable: {missing}")
    gate = pd.concat(
        [pd.read_csv(required[0]), pd.read_csv(required[2])], ignore_index=True
    )
    causal = pd.concat(
        [pd.read_csv(required[1]), pd.read_csv(required[3])], ignore_index=True
    )
    expected = [int(seed) for seed in study["development_seeds"]]
    if gate["seed"].astype(int).tolist() != expected:
        raise ValueError("closed T1-U7B gate rows do not match exposed seed order")
    if causal["seed"].astype(int).tolist() != expected:
        raise ValueError("closed T1-U7B causal rows do not match exposed seed order")
    return gate, causal


def paired_development_effects(
    baseline_gate: pd.DataFrame,
    candidate_gate: pd.DataFrame,
    baseline_causal: pd.DataFrame,
    candidate_causal: pd.DataFrame,
) -> pd.DataFrame:
    baseline = baseline_gate.merge(
        baseline_causal[
            [
                "seed",
                "causal_clean_accuracy",
                "counterfactual_accuracy",
                "counterfactual_prediction_flip_consistency",
            ]
        ],
        on="seed",
        how="left",
    )
    candidate = candidate_gate.merge(
        candidate_causal[
            [
                "seed",
                "causal_clean_accuracy",
                "counterfactual_accuracy",
                "counterfactual_prediction_flip_consistency",
            ]
        ],
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
        candidate[["seed", "passed", *columns]],
        on="seed",
        suffixes=("_t1_u7b", "_t1_u7c"),
        validate="one_to_one",
    )
    for column in columns:
        merged[f"delta_{column}"] = (
            merged[f"{column}_t1_u7c"] - merged[f"{column}_t1_u7b"]
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
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.8), constrained_layout=True)
    prefix_updates = int(study["closed_reference"]["expected_prefix_updates"])
    gate = study["development_gate"]
    styles = {"development": "-", "fresh_confirmation": "--"}

    for (phase, seed), group in training.groupby(["evaluation_phase", "seed"]):
        axes[0, 0].plot(
            group["step"],
            group["validation_final_accuracy"],
            linestyle=styles.get(str(phase), "-"),
            label=f"{phase}: {seed}",
        )
    axes[0, 0].axvline(
        prefix_updates,
        color="#777777",
        linestyle="--",
        label="auxiliary extension begins",
    )
    axes[0, 0].axhline(
        float(gate["minimum_final_accuracy"]), color="#c44e52", linestyle=":"
    )
    axes[0, 0].set(
        title="Fixed-schedule learning curves",
        xlabel="Training update",
        ylabel="Validation accuracy",
        ylim=(0.45, 1.01),
    )
    axes[0, 0].legend(frameon=False, fontsize=6.5, ncol=2)

    x = np.arange(len(effects))
    width = 0.36
    axes[0, 1].bar(
        x - width / 2,
        effects["final_accuracy_t1_u7b"],
        width,
        label="T1-U7B",
        color="#9ecae1",
    )
    axes[0, 1].bar(
        x + width / 2,
        effects["final_accuracy_t1_u7c"],
        width,
        label="T1-U7C",
        color="#3182bd",
    )
    axes[0, 1].axhline(
        float(gate["minimum_final_accuracy"]), color="#c44e52", linestyle=":"
    )
    minimum = min(
        float(effects["final_accuracy_t1_u7b"].min()),
        float(effects["final_accuracy_t1_u7c"].min()),
    )
    axes[0, 1].set(
        title="Paired clean-accuracy effect",
        ylabel="Fixed development accuracy",
        xticks=x,
        xticklabels=[str(int(seed)) for seed in effects["seed"]],
        ylim=(max(0.0, minimum - 0.05), 1.005),
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
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else [0.0]
        axes[1, 1].scatter(
            index + np.asarray(jitter), values, color="#222222", s=18
        )
    axes[1, 1].axhspan(
        float(gate["corruption_accuracy_minimum"]),
        float(gate["corruption_accuracy_maximum"]),
        color="#dddddd",
        alpha=0.5,
    )
    axes[1, 1].set(
        title="Causal controls after intervention",
        ylabel="Final accuracy",
        xticks=np.arange(len(metrics)),
        xticklabels=labels,
        ylim=(0.0, 1.01),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    fig.suptitle("T1-U7C intermediate-supervision stability repair", fontsize=14)
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
        output_dir / "t1_u7c_training_signal_repair.png",
    )

    smoke = str(study["experiment"]["id"]).endswith("-smoke")
    if smoke:
        status = "smoke_execution_path_complete"
    elif not development_passed:
        status = "failed_development_gate_new_confirmation_seeds_unopened"
    elif not fresh_passed:
        status = "passed_development_failed_new_confirmation"
    else:
        status = "passed_development_and_new_confirmation_ready_for_t1_u8"
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
        "training_signal_intervention": study["training_signal_intervention"],
        "configuration": {
            "development_seeds": development_seeds,
            "fresh_confirmation_seeds": study["fresh_confirmation_seeds"],
            "data": study["data"],
            "model": study["model"],
            "training": study["training"],
        },
        "gate_thresholds": study["development_gate"],
        "paired_development_effects_vs_t1_u7b": frame_records(effects),
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
