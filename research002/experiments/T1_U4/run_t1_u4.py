#!/usr/bin/env python3
"""Run the preregistered T1-U4 optimization-stability development study.

Candidate selection is validation-only. Independent data are evaluated only
for the selected intervention after it clears the fixed screening gate. This
runner contains no trajectory-probe fitting path.
"""

from __future__ import annotations

import argparse
import copy
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
    parser = argparse.ArgumentParser(description="Run T1-U4 optimization-stability study")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/t1_u4_optimization_stability.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/t1_u4"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_protocol(study: dict[str, Any]) -> None:
    candidates = study["factor_matrix"]
    ids = [str(candidate["id"]) for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique")
    references = [candidate for candidate in candidates if not bool(candidate["train"])]
    if len(references) > 1:
        raise ValueError("at most one closed reference is supported")

    total_updates: set[int] = set()
    auxiliary_areas: set[float] = set()
    for schedule_name, phases in study["schedules"].items():
        total = sum(int(phase["steps"]) for phase in phases)
        area = sum(
            int(phase["steps"]) * float(phase["auxiliary_loss_weight"])
            for phase in phases
        )
        total_updates.add(total)
        auxiliary_areas.add(round(area, 10))
        if any("data_stream_id" not in phase for phase in phases):
            raise ValueError(f"schedule {schedule_name} must pin every data_stream_id")
    if len(total_updates) != 1:
        raise ValueError("all schedules must have the same update budget")
    if len(auxiliary_areas) != 1:
        raise ValueError("all schedules must have equal integrated auxiliary weight")

    for candidate in candidates:
        if candidate["transition"] not in study["schedules"]:
            raise ValueError(f"unknown transition for {candidate['id']}")
        if float(candidate["residual_scale"]) <= 0.0:
            raise ValueError("residual_scale must be positive")


def candidate_config(
    study: dict[str, Any], candidate: dict[str, Any], seed: int
) -> dict[str, Any]:
    model = copy.deepcopy(study["base_model"])
    model["residual_scale"] = float(candidate["residual_scale"])
    training = copy.deepcopy(study["base_training"])
    training["curriculum"] = copy.deepcopy(study["schedules"][candidate["transition"]])
    return {
        "experiment": {
            **copy.deepcopy(study["experiment"]),
            "id": f"{study['experiment']['id']}-{candidate['id']}",
        },
        "hypotheses": copy.deepcopy(study["hypotheses"]),
        "seeds": [int(seed)],
        "data": copy.deepcopy(study["data"]),
        "model": model,
        "training": training,
        "probes": {"test_fraction": 0.4, "minimum_class_count": 30, "ridge_alpha": 1.0},
        "validity": copy.deepcopy(study["independent_validity"]),
    }


def annotate_history(
    history: pd.DataFrame, candidate: dict[str, Any]
) -> pd.DataFrame:
    result = history.copy()
    result.insert(0, "candidate_id", str(candidate["id"]))
    result.insert(1, "residual_scale", float(candidate["residual_scale"]))
    result.insert(2, "transition", str(candidate["transition"]))
    return result


def load_closed_reference(
    candidate: dict[str, Any], seeds: list[int]
) -> pd.DataFrame:
    path = Path(str(candidate["source_history"]))
    if not path.exists():
        raise FileNotFoundError(f"closed reference history not found: {path}")
    history = pd.read_csv(path)
    found_seeds = sorted(int(seed) for seed in history["seed"].unique())
    if found_seeds != sorted(seeds):
        raise ValueError(
            f"closed reference seeds {found_seeds} do not match preregistered seeds {seeds}"
        )
    return annotate_history(history, candidate)


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
        "trained_in_t1_u4": bool(candidate["train"]),
        "residual_scale": float(candidate["residual_scale"]),
        "transition": str(candidate["transition"]),
        "tail_mean_validation_accuracy": float(tail["validation_final_accuracy"].mean()),
        "tail_std_validation_accuracy": float(
            tail["validation_final_accuracy"].std(ddof=1) if len(tail) > 1 else 0.0
        ),
        "terminal_validation_accuracy": float(history.iloc[-1]["validation_final_accuracy"]),
        "best_validation_accuracy": float(history["validation_final_accuracy"].max()),
        "best_validation_step": int(
            history.loc[history["validation_final_accuracy"].idxmax(), "step"]
        ),
        "terminal_grad_norm": float(history.iloc[-1]["grad_norm"]),
        "maximum_logged_grad_norm": float(history["grad_norm"].max()),
    }


def aggregate_candidates(seed_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id, group in seed_results.groupby("candidate_id", sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "trained_in_t1_u4": bool(first["trained_in_t1_u4"]),
                "residual_scale": float(first["residual_scale"]),
                "transition": str(first["transition"]),
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


def rank_interventions(candidate_results: pd.DataFrame) -> pd.DataFrame:
    interventions = candidate_results[candidate_results["trained_in_t1_u4"]].copy()
    if interventions.empty:
        raise ValueError("at least one trainable intervention is required")
    return interventions.sort_values(
        by=[
            "minimum_seed_tail_mean_validation_accuracy",
            "mean_seed_tail_validation_accuracy",
            "minimum_terminal_validation_accuracy",
            "mean_terminal_validation_accuracy",
            "candidate_id",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def screening_decision(
    ranked: pd.DataFrame,
    candidate_results: pd.DataFrame,
    screening: dict[str, Any],
) -> dict[str, Any]:
    selected = ranked.iloc[0]
    references = candidate_results[~candidate_results["trained_in_t1_u4"]]
    if len(references) != 1:
        reference_primary = float("nan")
        primary_gain = float("nan")
        gain_passed = True
        reference_id = None
    else:
        reference = references.iloc[0]
        reference_primary = float(reference["minimum_seed_tail_mean_validation_accuracy"])
        primary_gain = (
            float(selected["minimum_seed_tail_mean_validation_accuracy"]) - reference_primary
        )
        gain_passed = primary_gain >= float(
            screening["minimum_primary_gain_over_reference"]
        )
        reference_id = str(reference["candidate_id"])

    minimum_passed = float(
        selected["minimum_seed_tail_mean_validation_accuracy"]
    ) >= float(screening["minimum_seed_tail_mean_validation_accuracy"])
    mean_passed = float(selected["mean_seed_tail_validation_accuracy"]) >= float(
        screening["minimum_mean_tail_validation_accuracy"]
    )
    passed = minimum_passed and mean_passed and gain_passed
    return {
        "selected_candidate_id": str(selected["candidate_id"]),
        "reference_candidate_id": reference_id,
        "selected_primary_metric": float(
            selected["minimum_seed_tail_mean_validation_accuracy"]
        ),
        "reference_primary_metric": reference_primary,
        "primary_gain_over_reference": primary_gain,
        "minimum_seed_tail_gate_passed": bool(minimum_passed),
        "mean_tail_gate_passed": bool(mean_passed),
        "gain_gate_passed": bool(gain_passed),
        "passed": bool(passed),
    }


def factorial_effects(candidate_results: pd.DataFrame) -> dict[str, float] | None:
    cells = {
        (float(row.residual_scale), str(row.transition)): float(
            row.mean_seed_tail_validation_accuracy
        )
        for row in candidate_results.itertuples()
    }
    required = {
        (1.0, "abrupt"),
        (0.5, "abrupt"),
        (1.0, "area_matched_taper"),
        (0.5, "area_matched_taper"),
    }
    if not required.issubset(cells):
        return None
    reference = cells[(1.0, "abrupt")]
    residual = cells[(0.5, "abrupt")]
    taper = cells[(1.0, "area_matched_taper")]
    combined = cells[(0.5, "area_matched_taper")]
    return {
        "residual_scale_main_effect": ((residual + combined) - (reference + taper)) / 2.0,
        "auxiliary_taper_main_effect": ((taper + combined) - (reference + residual)) / 2.0,
        "factor_interaction": combined - residual - taper + reference,
    }


def plot_screening(
    seed_results: pd.DataFrame,
    screening: dict[str, Any],
    output_path: Path,
) -> None:
    candidates = list(dict.fromkeys(seed_results["candidate_id"].tolist()))
    seeds = sorted(int(seed) for seed in seed_results["seed"].unique())
    x = np.arange(len(candidates), dtype=float)
    width = 0.8 / max(len(seeds), 1)
    fig, axis = plt.subplots(figsize=(10.0, 5.4), constrained_layout=True)
    for index, seed in enumerate(seeds):
        values = []
        for candidate_id in candidates:
            row = seed_results[
                (seed_results["candidate_id"] == candidate_id)
                & (seed_results["seed"] == seed)
            ]
            values.append(float(row.iloc[0]["tail_mean_validation_accuracy"]))
        offset = (index - (len(seeds) - 1) / 2.0) * width
        axis.bar(x + offset, values, width=width, label=f"seed {seed}")
    axis.axhline(
        float(screening["minimum_seed_tail_mean_validation_accuracy"]),
        color="#c44e52",
        linestyle="--",
        linewidth=1.2,
        label="screening minimum",
    )
    axis.set(
        title="T1-U4 optimization-stability screening",
        xlabel="Candidate",
        ylabel="Tail-mean validation accuracy",
        ylim=(0.45, 1.01),
    )
    axis.set_xticks(x, candidates, rotation=14, ha="right")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    study = load_config(config_path)
    validate_protocol(study)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seeds = [int(seed) for seed in study["seeds"]]
    tail_points = int(study["screening"]["tail_validation_points"])

    all_histories: list[pd.DataFrame] = []
    seed_rows: list[dict[str, Any]] = []
    trained: dict[tuple[str, int], tuple[Any, dict[str, np.ndarray]]] = {}

    for candidate in study["factor_matrix"]:
        candidate_id = str(candidate["id"])
        if not bool(candidate["train"]):
            print(f"REFERENCE candidate={candidate_id}", flush=True)
            candidate_history = load_closed_reference(candidate, seeds)
            all_histories.append(candidate_history)
            for seed in seeds:
                history = candidate_history[candidate_history["seed"] == seed]
                seed_rows.append(
                    summarize_seed_history(history, candidate, tail_points=tail_points)
                )
            continue

        for seed in seeds:
            print(f"CANDIDATE candidate={candidate_id} seed={seed}", flush=True)
            run_dir = output_dir / "candidates" / candidate_id / f"seed_{seed}"
            config = candidate_config(study, candidate, seed)
            model, history, independent_data = train_one_model(
                config, seed, run_dir / "checkpoints"
            )
            history_frame = annotate_history(pd.DataFrame(history), candidate)
            run_dir.mkdir(parents=True, exist_ok=True)
            history_frame.to_csv(run_dir / "training_history.csv", index=False)
            all_histories.append(history_frame)
            seed_rows.append(
                summarize_seed_history(history_frame, candidate, tail_points=tail_points)
            )
            trained[(candidate_id, seed)] = (model, independent_data)

    combined_history = pd.concat(all_histories, ignore_index=True)
    seed_results = pd.DataFrame(seed_rows)
    candidate_results = aggregate_candidates(seed_results)
    ranked = rank_interventions(candidate_results)
    selection = screening_decision(ranked, candidate_results, study["screening"])
    selected_id = str(selection["selected_candidate_id"])

    combined_history.to_csv(output_dir / "screening_training_history.csv", index=False)
    seed_results.to_csv(output_dir / "seed_screening_results.csv", index=False)
    candidate_results.to_csv(output_dir / "candidate_screening_results.csv", index=False)
    ranked.to_csv(output_dir / "intervention_ranking.csv", index=False)
    (output_dir / "selection_decision.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    plot_screening(
        seed_results, study["screening"], output_dir / "t1_u4_screening.png"
    )

    selected_candidate = next(
        candidate for candidate in study["factor_matrix"] if candidate["id"] == selected_id
    )
    selected_configuration = candidate_config(study, selected_candidate, seeds[0])
    selected_configuration["seeds"] = seeds
    with (output_dir / "selected_configuration.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(selected_configuration, handle, sort_keys=False)

    independent_rows: list[dict[str, Any]] = []
    independent_seed_rows: list[dict[str, Any]] = []
    if selection["passed"]:
        print(f"INDEPENDENT selected={selected_id}", flush=True)
        for seed in seeds:
            model, independent_data = trained[(selected_id, seed)]
            trajectory = extract_trajectory(
                model,
                independent_data["tokens"],
                independent_data["labels"],
                batch_size=int(study["base_training"]["validation_batch_size"]),
            )
            independent_dir = output_dir / "selected" / f"seed_{seed}" / "independent"
            independent_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                independent_dir / f"t1_u4_trajectory_seed_{seed}.npz", **trajectory
            )
            rows = trajectory_statistics(trajectory, seed)
            for row in rows:
                row["candidate_id"] = selected_id
            independent_rows.extend(rows)
            independent_seed_rows.append(
                {"candidate_id": selected_id, **seed_validity(rows, study["independent_validity"])}
            )

        pd.DataFrame(independent_rows).to_csv(
            output_dir / "independent_trajectory_statistics.csv", index=False
        )
        pd.DataFrame(independent_seed_rows).to_csv(
            output_dir / "independent_seed_validity.csv", index=False
        )
        passing_fraction = float(
            np.mean([bool(row["passed"]) for row in independent_seed_rows])
        )
        independent_accuracies = np.asarray(
            [float(row["final_accuracy"]) for row in independent_seed_rows],
            dtype=float,
        )
        independent_aggregate = {
            "mean_accuracy": float(np.mean(independent_accuracies)),
            "sample_std_accuracy": float(
                np.std(independent_accuracies, ddof=1)
                if len(independent_accuracies) > 1
                else 0.0
            ),
            "minimum_accuracy": float(np.min(independent_accuracies)),
            "maximum_accuracy": float(np.max(independent_accuracies)),
        }
        independent_passed = passing_fraction >= float(
            study["independent_validity"]["minimum_passing_seed_fraction"]
        )
        status = (
            "passed_development_gate_requires_fresh_seed_confirmation"
            if independent_passed
            else "failed_independent_stability_gate"
        )
    else:
        passing_fraction = None
        independent_aggregate = None
        independent_passed = False
        status = "failed_validation_screening_gate_independent_unopened"

    summary = {
        "experiment": study["experiment"],
        "status": status,
        "scientific_boundary": study["scientific_boundary"],
        "configuration": {
            "seeds": seeds,
            "data": study["data"],
            "base_model": study["base_model"],
            "base_training": study["base_training"],
            "schedules": study["schedules"],
            "factor_matrix": study["factor_matrix"],
        },
        "screening": {
            "thresholds": study["screening"],
            "decision": selection,
            "candidate_results": candidate_results.to_dict(orient="records"),
            "factorial_effects_on_mean_tail_accuracy": factorial_effects(candidate_results),
        },
        "independent_evaluation": {
            "opened": bool(selection["passed"]),
            "thresholds": study["independent_validity"],
            "passing_seed_fraction": passing_fraction,
            "passed": bool(independent_passed),
            "aggregate": independent_aggregate,
            "seeds": independent_seed_rows,
        },
        "sealed_evaluation": study["sealed_evaluation"],
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
