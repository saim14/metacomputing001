#!/usr/bin/env python3
"""Run the preregistered T1-U5 residual-scale calibration study.

All candidate models are evaluated on exposed development seeds and exposed
trajectory samples. Only a jointly eligible candidate may open the separately
generated task-sample holdout. This runner contains no probe-fitting path.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from run_t1_u2 import seed_validity, trajectory_statistics
from src.t1_experiment import (
    extract_trajectory,
    generate_task,
    train_one_model,
    write_environment_manifest,
)
from src.tiny_transformer import ModelConfig, TinyTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T1-U5 residual-scale calibration")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/t1_u5_residual_calibration.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/t1_u5"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_protocol(study: dict[str, Any]) -> None:
    candidates = study["candidates"]
    ids = [str(candidate["id"]) for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique")
    if not any(bool(candidate["train"]) for candidate in candidates):
        raise ValueError("at least one trainable candidate is required")
    if sum(not bool(candidate["train"]) for candidate in candidates) > 1:
        raise ValueError("at most one closed reference is supported")
    scales = [float(candidate["residual_scale"]) for candidate in candidates]
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("residual scales must be positive")
    trainable_scales = [
        float(candidate["residual_scale"])
        for candidate in candidates
        if bool(candidate["train"])
    ]
    if len(trainable_scales) != len(set(trainable_scales)):
        raise ValueError("trainable residual scales must be unique")

    curriculum = study["base_training"]["curriculum"]
    if any("data_stream_id" not in phase for phase in curriculum):
        raise ValueError("every curriculum phase must pin data_stream_id")
    if sum(int(phase["steps"]) for phase in curriculum) <= 0:
        raise ValueError("the fixed update budget must be positive")
    if int(study["sealed_sample"]["seed_offset"]) == 37:
        raise ValueError("sealed sample offset must differ from development offset 37")
    if int(study["execution"]["max_workers"]) < 1:
        raise ValueError("max_workers must be at least one")


def candidate_config(
    study: dict[str, Any], candidate: dict[str, Any], seed: int
) -> dict[str, Any]:
    model = copy.deepcopy(study["base_model"])
    model["residual_scale"] = float(candidate["residual_scale"])
    return {
        "experiment": {
            **copy.deepcopy(study["experiment"]),
            "id": f"{study['experiment']['id']}-{candidate['id']}",
        },
        "hypotheses": copy.deepcopy(study["hypotheses"]),
        "seeds": [int(seed)],
        "data": copy.deepcopy(study["data"]),
        "model": model,
        "training": copy.deepcopy(study["base_training"]),
        "probes": {"test_fraction": 0.4, "minimum_class_count": 30, "ridge_alpha": 1.0},
        "validity": copy.deepcopy(study["development_joint_gate"]),
    }


def annotate_history(
    history: pd.DataFrame, candidate: dict[str, Any]
) -> pd.DataFrame:
    result = history.copy()
    result.insert(0, "candidate_id", str(candidate["id"]))
    result.insert(1, "residual_scale", float(candidate["residual_scale"]))
    return result


def summarize_seed_history(
    history: pd.DataFrame,
    candidate: dict[str, Any],
    *,
    tail_points: int,
) -> dict[str, Any]:
    tail = history.tail(tail_points)
    return {
        "candidate_id": str(candidate["id"]),
        "seed": int(history.iloc[0]["seed"]),
        "trained_in_t1_u5": bool(candidate["train"]),
        "residual_scale": float(candidate["residual_scale"]),
        "tail_mean_validation_accuracy": float(tail["validation_final_accuracy"].mean()),
        "tail_std_validation_accuracy": float(
            tail["validation_final_accuracy"].std(ddof=1) if len(tail) > 1 else 0.0
        ),
        "terminal_validation_accuracy": float(history.iloc[-1]["validation_final_accuracy"]),
        "best_validation_accuracy": float(history["validation_final_accuracy"].max()),
        "best_validation_step": int(
            history.loc[history["validation_final_accuracy"].idxmax(), "step"]
        ),
    }


def load_closed_reference(
    candidate: dict[str, Any], seeds: list[int], thresholds: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    history_path = Path(str(candidate["source_history"]))
    statistics_path = Path(str(candidate["source_trajectory_statistics"]))
    if not history_path.exists() or not statistics_path.exists():
        raise FileNotFoundError("closed T1-U4 reference files are unavailable")
    source_id = str(candidate["source_candidate_id"])
    history = pd.read_csv(history_path)
    history = history[history["candidate_id"] == source_id].copy()
    found_seeds = sorted(int(seed) for seed in history["seed"].unique())
    if found_seeds != sorted(seeds):
        raise ValueError("closed reference history seeds do not match T1-U5 seeds")
    history = history.drop(columns=["candidate_id", "residual_scale", "transition"], errors="ignore")
    history = annotate_history(history, candidate)

    statistics = pd.read_csv(statistics_path)
    if "candidate_id" in statistics:
        statistics = statistics[statistics["candidate_id"] == source_id].copy()
    found_stat_seeds = sorted(int(seed) for seed in statistics["seed"].unique())
    if found_stat_seeds != sorted(seeds):
        raise ValueError("closed reference trajectory seeds do not match T1-U5 seeds")
    statistics["candidate_id"] = str(candidate["id"])
    statistics["residual_scale"] = float(candidate["residual_scale"])
    rows = statistics.to_dict(orient="records")
    validity: list[dict[str, Any]] = []
    for seed in seeds:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        validity.append(
            {
                "candidate_id": str(candidate["id"]),
                "residual_scale": float(candidate["residual_scale"]),
                **seed_validity(seed_rows, thresholds),
            }
        )
    return history, rows, validity


def train_candidate_job(
    study: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
    output_dir: str,
) -> dict[str, Any]:
    candidate_id = str(candidate["id"])
    run_dir = Path(output_dir) / "candidates" / candidate_id / f"seed_{seed}"
    config = candidate_config(study, candidate, seed)
    checkpoint = run_dir / "checkpoints" / f"t1_transformer_seed_{seed}.npz"
    history_path = run_dir / "training_history.csv"
    expected_steps = sum(
        int(phase["steps"]) for phase in study["base_training"]["curriculum"]
    )
    resumable = checkpoint.exists() and history_path.exists()
    if resumable:
        history_frame = pd.read_csv(history_path)
        resumable = (
            not history_frame.empty
            and int(history_frame.iloc[-1]["step"]) == expected_steps
        )

    if resumable:
        print(f"CANDIDATE_RESUME candidate={candidate_id} seed={seed}", flush=True)
        model = construct_model(config, seed, checkpoint)
        tokens, labels = generate_task(
            int(study["data"]["probe_samples"]),
            data_config=study["data"],
            rng=np.random.default_rng(seed * 100 + 37),
        )
        development_data = {"tokens": tokens, "labels": labels}
    else:
        print(f"CANDIDATE_START candidate={candidate_id} seed={seed}", flush=True)
        model, history, development_data = train_one_model(
            config, seed, run_dir / "checkpoints"
        )
        history_frame = annotate_history(pd.DataFrame(history), candidate)
        run_dir.mkdir(parents=True, exist_ok=True)
        history_frame.to_csv(history_path, index=False)
    trajectory = extract_trajectory(
        model,
        development_data["tokens"],
        development_data["labels"],
        batch_size=int(study["base_training"]["validation_batch_size"]),
    )
    trajectory_rows = trajectory_statistics(trajectory, seed)
    for row in trajectory_rows:
        row["candidate_id"] = candidate_id
        row["residual_scale"] = float(candidate["residual_scale"])
    validity = {
        "candidate_id": candidate_id,
        "residual_scale": float(candidate["residual_scale"]),
        **seed_validity(trajectory_rows, study["development_joint_gate"]),
    }
    print(
        f"CANDIDATE_DONE candidate={candidate_id} seed={seed} "
        f"accuracy={validity['final_accuracy']:.5f} "
        f"valid_layers={validity['valid_layer_count']}",
        flush=True,
    )
    return {
        "candidate_id": candidate_id,
        "seed": int(seed),
        "history": history_frame.to_dict(orient="records"),
        "seed_history": summarize_seed_history(
            history_frame,
            candidate,
            tail_points=int(study["validation_screen"]["tail_validation_points"]),
        ),
        "trajectory_rows": trajectory_rows,
        "validity": validity,
    }


def aggregate_validation(seed_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id, group in seed_results.groupby("candidate_id", sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "trained_in_t1_u5": bool(first["trained_in_t1_u5"]),
                "residual_scale": float(first["residual_scale"]),
                "minimum_seed_tail_mean_validation_accuracy": float(
                    group["tail_mean_validation_accuracy"].min()
                ),
                "mean_seed_tail_validation_accuracy": float(
                    group["tail_mean_validation_accuracy"].mean()
                ),
                "sample_std_seed_tail_validation_accuracy": float(
                    group["tail_mean_validation_accuracy"].std(ddof=1)
                    if len(group) > 1
                    else 0.0
                ),
                "minimum_terminal_validation_accuracy": float(
                    group["terminal_validation_accuracy"].min()
                ),
                "mean_terminal_validation_accuracy": float(
                    group["terminal_validation_accuracy"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def joint_candidate_results(
    validation: pd.DataFrame,
    validity: pd.DataFrame,
    study: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    validation_gate = study["validation_screen"]
    joint_gate = study["development_joint_gate"]
    candidates = {str(candidate["id"]): candidate for candidate in study["candidates"]}
    for candidate_id, group in validity.groupby("candidate_id", sort=False):
        val = validation[validation["candidate_id"] == candidate_id].iloc[0]
        candidate = candidates[str(candidate_id)]
        validation_passed = (
            float(val["minimum_seed_tail_mean_validation_accuracy"])
            >= float(validation_gate["minimum_seed_tail_mean_validation_accuracy"])
            and float(val["mean_seed_tail_validation_accuracy"])
            >= float(validation_gate["minimum_mean_tail_validation_accuracy"])
        )
        passing_fraction = float(group["passed"].astype(bool).mean())
        joint_passed = passing_fraction >= float(
            joint_gate["minimum_passing_seed_fraction"]
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "trained_in_t1_u5": bool(candidate["train"]),
                "residual_scale": float(candidate["residual_scale"]),
                "minimum_seed_tail_mean_validation_accuracy": float(
                    val["minimum_seed_tail_mean_validation_accuracy"]
                ),
                "mean_seed_tail_validation_accuracy": float(
                    val["mean_seed_tail_validation_accuracy"]
                ),
                "validation_screen_passed": bool(validation_passed),
                "minimum_development_final_accuracy": float(group["final_accuracy"].min()),
                "mean_development_final_accuracy": float(group["final_accuracy"].mean()),
                "sample_std_development_final_accuracy": float(
                    group["final_accuracy"].std(ddof=1) if len(group) > 1 else 0.0
                ),
                "minimum_valid_layer_count": int(group["valid_layer_count"].min()),
                "mean_valid_layer_count": float(group["valid_layer_count"].mean()),
                "passing_seed_fraction": passing_fraction,
                "development_joint_gate_passed": bool(joint_passed),
                "eligible": bool(candidate["train"] and validation_passed and joint_passed),
            }
        )
    return pd.DataFrame(rows)


def rank_candidates(joint: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "minimum_development_final_accuracy",
        "mean_development_final_accuracy",
        "minimum_valid_layer_count",
        "mean_valid_layer_count",
        "minimum_seed_tail_mean_validation_accuracy",
        "residual_scale",
        "candidate_id",
    ]
    ascending = [False, False, False, False, False, True, True]
    diagnostic = joint[joint["trained_in_t1_u5"]].sort_values(
        by=["eligible", "passing_seed_fraction", *columns],
        ascending=[False, False, *ascending],
        kind="mergesort",
    ).reset_index(drop=True)
    eligible = joint[joint["eligible"]].sort_values(
        by=columns, ascending=ascending, kind="mergesort"
    ).reset_index(drop=True)
    return eligible, diagnostic


def construct_model(config: dict[str, Any], seed: int, checkpoint: Path) -> TinyTransformer:
    data = config["data"]
    model_config = ModelConfig(
        vocab_size=int(data["key_count"]) + 3,
        sequence_length=2 + 2 * int(data["pair_count"]),
        **config["model"],
    )
    model = TinyTransformer(model_config, seed=seed)
    with np.load(checkpoint) as state:
        for parameter in model.parameters():
            if parameter.name not in state:
                raise KeyError(f"missing checkpoint parameter: {parameter.name}")
            parameter.data[...] = state[parameter.name]
    return model


def evaluate_sealed_sample(
    study: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = candidate_config(study, candidate, seed)
    checkpoint = (
        output_dir
        / "candidates"
        / str(candidate["id"])
        / f"seed_{seed}"
        / "checkpoints"
        / f"t1_transformer_seed_{seed}.npz"
    )
    model = construct_model(config, seed, checkpoint)
    offset = int(study["sealed_sample"]["seed_offset"])
    tokens, labels = generate_task(
        int(study["sealed_sample"]["samples"]),
        data_config=study["data"],
        rng=np.random.default_rng(seed * 100 + offset),
    )
    trajectory = extract_trajectory(
        model,
        tokens,
        labels,
        batch_size=int(study["base_training"]["validation_batch_size"]),
    )
    sealed_dir = output_dir / "selected" / f"seed_{seed}" / "sealed_sample"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sealed_dir / f"t1_u5_sealed_trajectory_seed_{seed}.npz", **trajectory
    )
    rows = trajectory_statistics(trajectory, seed)
    for row in rows:
        row["candidate_id"] = str(candidate["id"])
        row["residual_scale"] = float(candidate["residual_scale"])
    validity = {
        "candidate_id": str(candidate["id"]),
        "residual_scale": float(candidate["residual_scale"]),
        **seed_validity(rows, study["sealed_sample"]),
    }
    return rows, validity


def plot_calibration(
    seed_validation: pd.DataFrame,
    development_validity: pd.DataFrame,
    study: dict[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    seed_colors = {seed: color for seed, color in zip(
        sorted(seed_validation["seed"].unique()), ["#4c78a8", "#f58518", "#54a24b"]
    )}
    for seed, group in seed_validation.groupby("seed"):
        group = group.sort_values("residual_scale")
        axes[0].plot(
            group["residual_scale"], group["tail_mean_validation_accuracy"],
            marker="o", label=f"seed {int(seed)}", color=seed_colors[seed]
        )
    axes[0].axhline(
        float(study["validation_screen"]["minimum_seed_tail_mean_validation_accuracy"]),
        color="#c44e52", linestyle="--", linewidth=1.1
    )
    axes[0].set(title="Learning stability", xlabel="Residual scale", ylabel="Tail validation accuracy", ylim=(0.45, 1.01))

    for seed, group in development_validity.groupby("seed"):
        group = group.sort_values("residual_scale")
        axes[1].plot(
            group["residual_scale"], group["final_accuracy"],
            marker="o", label=f"seed {int(seed)}", color=seed_colors[seed]
        )
        axes[2].plot(
            group["residual_scale"], group["valid_layer_count"],
            marker="o", label=f"seed {int(seed)}", color=seed_colors[seed]
        )
    axes[1].axhline(
        float(study["development_joint_gate"]["minimum_final_accuracy"]),
        color="#c44e52", linestyle="--", linewidth=1.1
    )
    axes[1].set(title="Development accuracy", xlabel="Residual scale", ylabel="Final accuracy", ylim=(0.45, 1.01))
    axes[2].axhline(
        int(study["development_joint_gate"]["minimum_valid_layers_per_seed"]),
        color="#c44e52", linestyle="--", linewidth=1.1
    )
    axes[2].set(title="Observable computation window", xlabel="Residual scale", ylabel="Valid intermediate steps")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("T1-U5 residual-scale calibration")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    study = load_config(config_path)
    validate_protocol(study)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seeds = [int(seed) for seed in study["seeds"]]
    tail_points = int(study["validation_screen"]["tail_validation_points"])

    histories: list[pd.DataFrame] = []
    seed_history_rows: list[dict[str, Any]] = []
    development_rows: list[dict[str, Any]] = []
    development_validity_rows: list[dict[str, Any]] = []

    for candidate in study["candidates"]:
        if bool(candidate["train"]):
            continue
        print(f"REFERENCE candidate={candidate['id']}", flush=True)
        history, rows, validity = load_closed_reference(
            candidate, seeds, study["development_joint_gate"]
        )
        histories.append(history)
        development_rows.extend(rows)
        development_validity_rows.extend(validity)
        for seed in seeds:
            seed_history = history[history["seed"] == seed]
            seed_history_rows.append(
                summarize_seed_history(seed_history, candidate, tail_points=tail_points)
            )

    jobs = [
        (candidate, seed)
        for candidate in study["candidates"]
        if bool(candidate["train"])
        for seed in seeds
    ]
    max_workers = min(int(study["execution"]["max_workers"]), len(jobs))
    if max_workers == 1:
        results = [
            train_candidate_job(study, candidate, seed, str(output_dir))
            for candidate, seed in jobs
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    train_candidate_job, study, candidate, seed, str(output_dir)
                ): (str(candidate["id"]), seed)
                for candidate, seed in jobs
            }
            for future in as_completed(futures):
                candidate_id, seed = futures[future]
                result = future.result()
                results.append(result)
                print(f"COLLECTED candidate={candidate_id} seed={seed}", flush=True)

    for result in sorted(results, key=lambda item: (item["candidate_id"], item["seed"])):
        histories.append(pd.DataFrame(result["history"]))
        seed_history_rows.append(result["seed_history"])
        development_rows.extend(result["trajectory_rows"])
        development_validity_rows.append(result["validity"])

    history_frame = pd.concat(histories, ignore_index=True)
    seed_history_frame = pd.DataFrame(seed_history_rows)
    development_frame = pd.DataFrame(development_rows)
    development_validity_frame = pd.DataFrame(development_validity_rows)
    validation_frame = aggregate_validation(seed_history_frame)
    joint_frame = joint_candidate_results(
        validation_frame, development_validity_frame, study
    )
    eligible, diagnostic = rank_candidates(joint_frame)

    history_frame.to_csv(output_dir / "calibration_training_history.csv", index=False)
    seed_history_frame.to_csv(output_dir / "validation_seed_results.csv", index=False)
    validation_frame.to_csv(output_dir / "validation_candidate_summary.csv", index=False)
    development_frame.to_csv(output_dir / "development_trajectory_statistics.csv", index=False)
    development_validity_frame.to_csv(output_dir / "development_seed_validity.csv", index=False)
    joint_frame.to_csv(output_dir / "candidate_joint_results.csv", index=False)
    diagnostic.to_csv(output_dir / "candidate_diagnostic_ranking.csv", index=False)
    plot_calibration(
        seed_history_frame,
        development_validity_frame,
        study,
        output_dir / "t1_u5_residual_calibration.png",
    )

    selected_id: str | None = None
    sealed_opened = False
    sealed_rows: list[dict[str, Any]] = []
    sealed_validity_rows: list[dict[str, Any]] = []
    if not eligible.empty:
        selected_id = str(eligible.iloc[0]["candidate_id"])
        selected_candidate = next(
            candidate for candidate in study["candidates"] if candidate["id"] == selected_id
        )
        selected_configuration = candidate_config(study, selected_candidate, seeds[0])
        selected_configuration["seeds"] = seeds
        with (output_dir / "selected_configuration.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(selected_configuration, handle, sort_keys=False)

        sealed_opened = True
        print(f"SEALED_SAMPLE selected={selected_id}", flush=True)
        for seed in seeds:
            rows, validity = evaluate_sealed_sample(
                study, selected_candidate, seed, output_dir
            )
            sealed_rows.extend(rows)
            sealed_validity_rows.append(validity)
            print(
                f"SEALED_RESULT seed={seed} accuracy={validity['final_accuracy']:.5f} "
                f"valid_layers={validity['valid_layer_count']} passed={validity['passed']}",
                flush=True,
            )
        pd.DataFrame(sealed_rows).to_csv(
            output_dir / "sealed_trajectory_statistics.csv", index=False
        )
        pd.DataFrame(sealed_validity_rows).to_csv(
            output_dir / "sealed_seed_validity.csv", index=False
        )
        sealed_passing_fraction = float(
            np.mean([bool(row["passed"]) for row in sealed_validity_rows])
        )
        sealed_passed = sealed_passing_fraction >= float(
            study["sealed_sample"]["minimum_passing_seed_fraction"]
        )
        sealed_accuracies = np.asarray(
            [float(row["final_accuracy"]) for row in sealed_validity_rows]
        )
        sealed_aggregate: dict[str, Any] | None = {
            "mean_accuracy": float(np.mean(sealed_accuracies)),
            "sample_std_accuracy": float(
                np.std(sealed_accuracies, ddof=1) if len(sealed_accuracies) > 1 else 0.0
            ),
            "minimum_accuracy": float(np.min(sealed_accuracies)),
            "maximum_accuracy": float(np.max(sealed_accuracies)),
        }
        status = (
            "passed_development_and_sample_holdout_requires_fresh_seed_confirmation"
            if sealed_passed
            else "failed_sealed_sample_joint_gate"
        )
    else:
        sealed_passing_fraction = None
        sealed_passed = False
        sealed_aggregate = None
        status = "failed_development_joint_gate_sealed_sample_unopened"

    selection = {
        "selected_candidate_id": selected_id,
        "eligible_candidate_ids": eligible["candidate_id"].tolist(),
        "eligible_candidate_count": int(len(eligible)),
        "sealed_sample_opened": bool(sealed_opened),
        "rule": study["selection"],
    }
    (output_dir / "selection_decision.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    summary = {
        "experiment": study["experiment"],
        "status": status,
        "scientific_boundary": study["scientific_boundary"],
        "configuration": {
            "seeds": seeds,
            "data": study["data"],
            "base_model": study["base_model"],
            "base_training": study["base_training"],
            "candidates": study["candidates"],
        },
        "validation_screen": {
            "thresholds": study["validation_screen"],
            "candidates": validation_frame.to_dict(orient="records"),
        },
        "development_joint_gate": {
            "thresholds": study["development_joint_gate"],
            "candidates": joint_frame.to_dict(orient="records"),
            "seeds": development_validity_frame.to_dict(orient="records"),
        },
        "selection": selection,
        "sealed_sample": {
            "opened": bool(sealed_opened),
            "thresholds_and_rule": study["sealed_sample"],
            "passing_seed_fraction": sealed_passing_fraction,
            "passed": bool(sealed_passed),
            "aggregate": sealed_aggregate,
            "seeds": sealed_validity_rows,
        },
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
