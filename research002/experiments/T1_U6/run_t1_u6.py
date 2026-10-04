#!/usr/bin/env python3
"""Run T1-U6: frozen-weight, inference-only residual pacing.

The T1-U4 learned weights remain fixed. Only the residual scale assigned to
each recurrent refinement step changes. No training or probe fitting occurs.
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
from src.t1_experiment import extract_trajectory, generate_task, write_environment_manifest
from src.tiny_transformer import ModelConfig, TinyTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T1-U6 frozen residual pacing")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/t1_u6_frozen_pacing.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/t1_u6"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_protocol(study: dict[str, Any]) -> None:
    n_layers = int(study["model"]["n_layers"])
    schedules = study["schedules"]
    if set(schedules) != {"constant", "back_loaded", "front_loaded"}:
        raise ValueError("T1-U6 requires constant, back_loaded, and front_loaded")
    required_sum = float(study["area_matching"]["required_sum"])
    tolerance = float(study["area_matching"]["tolerance"])
    for schedule_id, schedule in schedules.items():
        values = [float(value) for value in schedule["values"]]
        if len(values) != n_layers:
            raise ValueError(f"{schedule_id} must contain one scale per layer")
        if any(value <= 0.0 for value in values):
            raise ValueError(f"{schedule_id} scales must be positive")
        if not np.isclose(sum(values), required_sum, atol=tolerance, rtol=0.0):
            raise ValueError(f"{schedule_id} is not area matched")
    back = [float(value) for value in schedules["back_loaded"]["values"]]
    front = [float(value) for value in schedules["front_loaded"]["values"]]
    if sorted(back) != sorted(front) or front != list(reversed(back)):
        raise ValueError("ordered interventions must be exact reverse multisets")
    if not all(left < right for left, right in zip(back[:-1], back[1:])):
        raise ValueError("back_loaded must be strictly increasing")
    if int(study["sealed_sample"]["seed_offset"]) == int(
        study["data"]["development_seed_offset"]
    ):
        raise ValueError("sealed and development sample offsets must differ")


def model_config(
    study: dict[str, Any], schedule_values: list[float]
) -> ModelConfig:
    data = study["data"]
    model = copy.deepcopy(study["model"])
    model["residual_scale_schedule"] = tuple(float(value) for value in schedule_values)
    return ModelConfig(
        vocab_size=int(data["key_count"]) + 3,
        sequence_length=2 + 2 * int(data["pair_count"]),
        **model,
    )


def load_frozen_model(
    study: dict[str, Any], schedule_values: list[float], seed: int
) -> TinyTransformer:
    model = TinyTransformer(model_config(study, schedule_values), seed=seed)
    checkpoint = Path(
        str(study["checkpoint_source"]["pattern"]).format(seed=seed)
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"frozen checkpoint not found: {checkpoint}")
    with np.load(checkpoint) as state:
        for parameter in model.parameters():
            if parameter.name not in state:
                raise KeyError(f"checkpoint missing parameter: {parameter.name}")
            parameter.data[...] = state[parameter.name]
    return model


def make_sample(
    study: dict[str, Any], *, seed: int, offset: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    return generate_task(
        samples,
        data_config=study["data"],
        rng=np.random.default_rng(seed * 100 + offset),
    )


def state_update_statistics(
    trajectory: dict[str, np.ndarray], schedule_id: str, seed: int
) -> list[dict[str, Any]]:
    states = trajectory["cls_states"]
    updates = states[:, 1:, :] - states[:, :-1, :]
    norms = np.linalg.norm(updates, axis=-1)
    previous_norms = np.maximum(np.linalg.norm(states[:, :-1, :], axis=-1), 1e-8)
    relative = norms / previous_norms
    rows: list[dict[str, Any]] = []
    for index in range(norms.shape[1]):
        rows.append(
            {
                "schedule_id": schedule_id,
                "seed": int(seed),
                "step": int(index + 1),
                "mean_cls_update_norm": float(np.mean(norms[:, index])),
                "median_cls_update_norm": float(np.median(norms[:, index])),
                "p90_cls_update_norm": float(np.quantile(norms[:, index], 0.90)),
                "mean_relative_cls_update_norm": float(np.mean(relative[:, index])),
            }
        )
    return rows


def evaluate_schedule(
    study: dict[str, Any],
    schedule_id: str,
    schedule: dict[str, Any],
    seed: int,
    *,
    offset: int,
    samples: int,
    thresholds: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    values = [float(value) for value in schedule["values"]]
    model = load_frozen_model(study, values, seed)
    tokens, labels = make_sample(study, seed=seed, offset=offset, samples=samples)
    trajectory = extract_trajectory(model, tokens, labels, batch_size=600)
    rows = trajectory_statistics(trajectory, seed)
    for row in rows:
        row["schedule_id"] = schedule_id
    validity = {
        "schedule_id": schedule_id,
        "selectable": bool(schedule["selectable"]),
        **seed_validity(rows, thresholds),
    }
    update_rows = state_update_statistics(trajectory, schedule_id, seed)
    return trajectory, rows, validity, update_rows


def verify_closed_reference(
    study: dict[str, Any], observed: pd.DataFrame
) -> dict[str, Any]:
    reference_path = study["checkpoint_source"].get("reference_statistics")
    if not reference_path:
        return {"performed": False, "passed": None, "maximum_absolute_error": None}
    expected = pd.read_csv(Path(str(reference_path)))
    source_id = study["checkpoint_source"].get("reference_candidate_id")
    if source_id is not None and "candidate_id" in expected:
        expected = expected[expected["candidate_id"] == source_id].copy()
    observed = observed[observed["schedule_id"] == "constant"].copy()
    keys = ["seed", "layer"]
    numeric = [
        "current_accuracy",
        "final_accuracy",
        "future_flip_rate",
        "future_correction_rate",
        "mean_delta_nll",
        "median_delta_nll",
    ]
    merged = expected[keys + numeric].merge(
        observed[keys + numeric], on=keys, suffixes=("_expected", "_observed")
    )
    if len(merged) != len(expected) or len(merged) != len(observed):
        raise ValueError("closed-reference rows do not align")
    errors = [
        np.max(np.abs(merged[f"{column}_expected"] - merged[f"{column}_observed"]))
        for column in numeric
    ]
    maximum_error = float(max(errors, default=0.0))
    passed = maximum_error <= 1e-7
    if not passed:
        raise AssertionError(f"constant schedule failed closed-reference check: {maximum_error}")
    return {
        "performed": True,
        "passed": True,
        "expected_source": str(reference_path),
        "rows_compared": int(len(merged)),
        "maximum_absolute_error": maximum_error,
        "tolerance": 1e-7,
    }


def aggregate_schedules(
    validity: pd.DataFrame, study: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    minimum_fraction = float(study["joint_gate"]["minimum_passing_seed_fraction"])
    for schedule_id, group in validity.groupby("schedule_id", sort=False):
        schedule = study["schedules"][schedule_id]
        passing_fraction = float(group["passed"].astype(bool).mean())
        gate_passed = passing_fraction >= minimum_fraction
        rows.append(
            {
                "schedule_id": schedule_id,
                "selectable": bool(schedule["selectable"]),
                "schedule_values": json.dumps(schedule["values"]),
                "scale_sum": float(sum(float(value) for value in schedule["values"])),
                "minimum_development_final_accuracy": float(group["final_accuracy"].min()),
                "mean_development_final_accuracy": float(group["final_accuracy"].mean()),
                "sample_std_development_final_accuracy": float(
                    group["final_accuracy"].std(ddof=1) if len(group) > 1 else 0.0
                ),
                "minimum_valid_layer_count": int(group["valid_layer_count"].min()),
                "mean_valid_layer_count": float(group["valid_layer_count"].mean()),
                "passing_seed_fraction": passing_fraction,
                "development_joint_gate_passed": bool(gate_passed),
                "eligible": bool(schedule["selectable"] and gate_passed),
            }
        )
    return pd.DataFrame(rows)


def rank_schedules(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "minimum_development_final_accuracy",
        "mean_development_final_accuracy",
        "minimum_valid_layer_count",
        "mean_valid_layer_count",
        "schedule_id",
    ]
    ascending = [False, False, False, False, True]
    diagnostic = summary[summary["selectable"]].sort_values(
        by=["eligible", "passing_seed_fraction", *columns],
        ascending=[False, False, *ascending],
        kind="mergesort",
    ).reset_index(drop=True)
    eligible = summary[summary["eligible"]].sort_values(
        by=columns, ascending=ascending, kind="mergesort"
    ).reset_index(drop=True)
    return eligible, diagnostic


def paired_order_effects(validity: pd.DataFrame) -> pd.DataFrame:
    back = validity[validity["schedule_id"] == "back_loaded"].set_index("seed")
    front = validity[validity["schedule_id"] == "front_loaded"].set_index("seed")
    seeds = sorted(set(back.index) & set(front.index))
    rows = []
    for seed in seeds:
        rows.append(
            {
                "seed": int(seed),
                "back_minus_front_final_accuracy": float(
                    back.loc[seed, "final_accuracy"] - front.loc[seed, "final_accuracy"]
                ),
                "back_minus_front_valid_layer_count": int(
                    back.loc[seed, "valid_layer_count"]
                    - front.loc[seed, "valid_layer_count"]
                ),
                "back_passed": bool(back.loc[seed, "passed"]),
                "front_passed": bool(front.loc[seed, "passed"]),
            }
        )
    return pd.DataFrame(rows)


def plot_results(
    validity: pd.DataFrame,
    updates: pd.DataFrame,
    study: dict[str, Any],
    output_path: Path,
) -> None:
    colors = {"constant": "#4c78a8", "back_loaded": "#54a24b", "front_loaded": "#e45756"}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    schedule_ids = list(study["schedules"])
    for schedule_id in schedule_ids:
        values = study["schedules"][schedule_id]["values"]
        axes[0, 0].plot(range(1, len(values) + 1), values, marker="o", label=schedule_id.replace("_", " "), color=colors[schedule_id])
    axes[0, 0].set(title="Preregistered residual schedules", xlabel="Refinement step", ylabel="Residual scale")

    x = np.arange(len(schedule_ids), dtype=float)
    seeds = sorted(int(seed) for seed in validity["seed"].unique())
    width = 0.18
    for index, seed in enumerate(seeds):
        group = validity[validity["seed"] == seed].set_index("schedule_id")
        values = [float(group.loc[schedule_id, "final_accuracy"]) for schedule_id in schedule_ids]
        axes[0, 1].bar(x + (index - (len(seeds) - 1) / 2) * width, values, width=width, label=f"seed {seed}")
    axes[0, 1].axhline(float(study["joint_gate"]["minimum_final_accuracy"]), color="#222222", linestyle="--", linewidth=1.0)
    axes[0, 1].set(title="Frozen-weight final accuracy", ylabel="Accuracy", ylim=(0.45, 1.01))
    axes[0, 1].set_xticks(x, [value.replace("_", " ") for value in schedule_ids], rotation=12)
    axes[0, 1].legend(frameon=False, fontsize=8)

    for index, seed in enumerate(seeds):
        group = validity[validity["seed"] == seed].set_index("schedule_id")
        values = [int(group.loc[schedule_id, "valid_layer_count"]) for schedule_id in schedule_ids]
        axes[1, 0].bar(x + (index - (len(seeds) - 1) / 2) * width, values, width=width, label=f"seed {seed}")
    axes[1, 0].axhline(int(study["joint_gate"]["minimum_valid_layers_per_seed"]), color="#222222", linestyle="--", linewidth=1.0)
    axes[1, 0].set(title="Observable computation window", ylabel="Valid intermediate steps")
    axes[1, 0].set_xticks(x, [value.replace("_", " ") for value in schedule_ids], rotation=12)

    grouped = updates.groupby(["schedule_id", "step"])["mean_cls_update_norm"].mean()
    for schedule_id in schedule_ids:
        values = grouped.loc[schedule_id]
        axes[1, 1].plot(values.index, values.values, marker="o", label=schedule_id.replace("_", " "), color=colors[schedule_id])
    axes[1, 1].set(title="Realized CLS-state movement", xlabel="Refinement step", ylabel="Mean update norm")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("T1-U6 frozen-weight residual pacing")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    study = load_config(config_path)
    validate_protocol(study)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seeds = [int(seed) for seed in study["seeds"]]
    development_offset = int(study["data"]["development_seed_offset"])
    development_samples = int(study["data"]["development_samples"])

    development_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    for schedule_id, schedule in study["schedules"].items():
        for seed in seeds:
            print(f"DEVELOPMENT schedule={schedule_id} seed={seed}", flush=True)
            _, rows, validity, updates = evaluate_schedule(
                study,
                schedule_id,
                schedule,
                seed,
                offset=development_offset,
                samples=development_samples,
                thresholds=study["joint_gate"],
            )
            development_rows.extend(rows)
            validity_rows.append(validity)
            update_rows.extend(updates)
            print(
                f"DEVELOPMENT_RESULT schedule={schedule_id} seed={seed} "
                f"accuracy={validity['final_accuracy']:.5f} "
                f"valid_layers={validity['valid_layer_count']} passed={validity['passed']}",
                flush=True,
            )

    development_frame = pd.DataFrame(development_rows)
    validity_frame = pd.DataFrame(validity_rows)
    update_frame = pd.DataFrame(update_rows)
    reference_check = verify_closed_reference(study, development_frame)
    schedule_frame = aggregate_schedules(validity_frame, study)
    eligible, diagnostic = rank_schedules(schedule_frame)
    order_effect_frame = paired_order_effects(validity_frame)

    development_frame.to_csv(output_dir / "development_trajectory_statistics.csv", index=False)
    validity_frame.to_csv(output_dir / "development_seed_validity.csv", index=False)
    update_frame.to_csv(output_dir / "state_update_statistics.csv", index=False)
    schedule_frame.to_csv(output_dir / "schedule_joint_results.csv", index=False)
    diagnostic.to_csv(output_dir / "schedule_diagnostic_ranking.csv", index=False)
    order_effect_frame.to_csv(output_dir / "paired_order_effects.csv", index=False)
    (output_dir / "reference_verification.json").write_text(
        json.dumps(reference_check, indent=2), encoding="utf-8"
    )
    plot_results(validity_frame, update_frame, study, output_dir / "t1_u6_frozen_pacing.png")

    selected_id: str | None = None
    sealed_opened = False
    sealed_rows: list[dict[str, Any]] = []
    sealed_validity_rows: list[dict[str, Any]] = []
    if not eligible.empty:
        selected_id = str(eligible.iloc[0]["schedule_id"])
        selected_schedule = study["schedules"][selected_id]
        with (output_dir / "selected_schedule.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {"schedule_id": selected_id, **selected_schedule}, handle, sort_keys=False
            )
        sealed_opened = True
        print(f"SEALED_SAMPLE schedule={selected_id}", flush=True)
        for seed in seeds:
            trajectory, rows, validity, _ = evaluate_schedule(
                study,
                selected_id,
                selected_schedule,
                seed,
                offset=int(study["sealed_sample"]["seed_offset"]),
                samples=int(study["sealed_sample"]["samples"]),
                thresholds=study["sealed_sample"],
            )
            sealed_rows.extend(rows)
            sealed_validity_rows.append(validity)
            sealed_dir = output_dir / "selected" / f"seed_{seed}" / "sealed_sample"
            sealed_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                sealed_dir / f"t1_u6_sealed_trajectory_seed_{seed}.npz", **trajectory
            )
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
        sealed_fraction = float(
            np.mean([bool(row["passed"]) for row in sealed_validity_rows])
        )
        sealed_passed = sealed_fraction >= float(
            study["sealed_sample"]["minimum_passing_seed_fraction"]
        )
        accuracies = np.asarray(
            [float(row["final_accuracy"]) for row in sealed_validity_rows]
        )
        sealed_aggregate: dict[str, Any] | None = {
            "mean_accuracy": float(np.mean(accuracies)),
            "sample_std_accuracy": float(
                np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
            ),
            "minimum_accuracy": float(np.min(accuracies)),
            "maximum_accuracy": float(np.max(accuracies)),
        }
        status = (
            "passed_frozen_pacing_and_sample_holdout_requires_fresh_seed_confirmation"
            if sealed_passed
            else "failed_sealed_sample_joint_gate"
        )
    else:
        sealed_fraction = None
        sealed_passed = False
        sealed_aggregate = None
        status = "failed_development_joint_gate_sealed_sample_unopened"

    selection = {
        "selected_schedule_id": selected_id,
        "eligible_schedule_ids": eligible["schedule_id"].tolist(),
        "eligible_schedule_count": int(len(eligible)),
        "sealed_sample_opened": bool(sealed_opened),
        "rule": study["selection"],
    }
    (output_dir / "selection_decision.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    order_effect_summary = {
        "mean_back_minus_front_final_accuracy": float(
            order_effect_frame["back_minus_front_final_accuracy"].mean()
        ),
        "mean_back_minus_front_valid_layer_count": float(
            order_effect_frame["back_minus_front_valid_layer_count"].mean()
        ),
        "back_better_accuracy_seed_count": int(
            (order_effect_frame["back_minus_front_final_accuracy"] > 0).sum()
        ),
        "back_better_validity_seed_count": int(
            (order_effect_frame["back_minus_front_valid_layer_count"] > 0).sum()
        ),
    }
    summary = {
        "experiment": study["experiment"],
        "status": status,
        "scientific_boundary": study["scientific_boundary"],
        "configuration": {
            "seeds": seeds,
            "data": study["data"],
            "model": study["model"],
            "checkpoint_source": study["checkpoint_source"],
            "schedules": study["schedules"],
            "area_matching": study["area_matching"],
        },
        "reference_verification": reference_check,
        "development_joint_gate": {
            "thresholds": study["joint_gate"],
            "schedules": schedule_frame.to_dict(orient="records"),
            "seeds": validity_frame.to_dict(orient="records"),
        },
        "paired_order_effects": {
            "summary": order_effect_summary,
            "seeds": order_effect_frame.to_dict(orient="records"),
        },
        "selection": selection,
        "sealed_sample": {
            "opened": bool(sealed_opened),
            "thresholds_and_rule": study["sealed_sample"],
            "passing_seed_fraction": sealed_fraction,
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
