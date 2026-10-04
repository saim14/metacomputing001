#!/usr/bin/env python3
"""Run the preregistered T1-U2 bounded learning study.

Candidate models are ranked only by validation accuracy.  Independent probe
data remain sealed until the selected candidate clears the advancement gate.
No trajectory-history probes are fit in this study.
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

from src.t1_experiment import extract_trajectory, train_one_model, write_environment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T1-U2 bounded learning study")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/t1_u2_learning_study.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/t1_u2"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate_config(study: dict[str, Any], candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "experiment": {
            "id": f"T1-U2-{candidate['id']}",
            "name": study["experiment"]["name"],
            "version": study["experiment"]["version"],
            "research_object": study["experiment"]["research_object"],
        },
        "hypotheses": study["hypotheses"],
        "seeds": [int(seed)],
        "data": copy.deepcopy(study["data"]),
        "model": {**copy.deepcopy(study["base_model"]), **copy.deepcopy(candidate["model"])},
        "training": copy.deepcopy(study["base_training"]),
        "probes": {
            "test_fraction": 0.4,
            "minimum_class_count": 30,
            "ridge_alpha": 1.0,
        },
        "validity": copy.deepcopy(study["independent_validity"]),
    }


def unique_parameter_count(model: Any) -> int:
    unique = {id(parameter): parameter for parameter in model.parameters()}
    return int(sum(parameter.data.size for parameter in unique.values()))


def validation_record(
    candidate: dict[str, Any],
    model: Any,
    history: list[dict[str, Any]],
    tail_checkpoints: int,
) -> dict[str, Any]:
    frame = pd.DataFrame(history)
    tail = frame.tail(tail_checkpoints)
    return {
        "candidate_id": candidate["id"],
        "rationale": candidate["rationale"],
        "d_model": int(model.config.d_model),
        "n_heads": int(model.config.n_heads),
        "d_ff": int(model.config.d_ff),
        "n_layers": int(model.config.n_layers),
        "use_step_embeddings": bool(model.config.use_step_embeddings),
        "unique_parameters": unique_parameter_count(model),
        "tail_mean_validation_accuracy": float(tail["validation_final_accuracy"].mean()),
        "terminal_validation_accuracy": float(frame.iloc[-1]["validation_final_accuracy"]),
        "best_validation_accuracy": float(frame["validation_final_accuracy"].max()),
        "best_validation_step": int(
            frame.loc[frame["validation_final_accuracy"].idxmax(), "step"]
        ),
    }


def trajectory_statistics(trajectory: dict[str, np.ndarray], seed: int) -> list[dict[str, Any]]:
    probabilities = trajectory["probabilities"]
    labels = trajectory["labels"]
    predictions = np.argmax(probabilities, axis=-1)
    final_layer = probabilities.shape[1] - 1
    final_prediction = predictions[:, final_layer]
    rows: list[dict[str, Any]] = []
    for layer in range(1, final_layer):
        current_prediction = predictions[:, layer]
        future_flip = current_prediction != final_prediction
        future_correction = (current_prediction != labels) & (final_prediction == labels)
        current_nll = -np.log(
            np.clip(probabilities[np.arange(len(labels)), layer, labels], 1e-8, 1.0)
        )
        final_nll = -np.log(
            np.clip(probabilities[np.arange(len(labels)), final_layer, labels], 1e-8, 1.0)
        )
        rows.append(
            {
                "seed": int(seed),
                "layer": int(layer),
                "samples": int(len(labels)),
                "current_accuracy": float(np.mean(current_prediction == labels)),
                "final_accuracy": float(np.mean(final_prediction == labels)),
                "future_flip_rate": float(np.mean(future_flip)),
                "future_correction_rate": float(np.mean(future_correction)),
                "mean_delta_nll": float(np.mean(current_nll - final_nll)),
                "median_delta_nll": float(np.median(current_nll - final_nll)),
            }
        )
    return rows


def seed_validity(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    final_accuracy = float(rows[0]["final_accuracy"])
    minimum_flip = float(thresholds["minimum_future_flip_rate"])
    maximum_flip = float(thresholds["maximum_future_flip_rate"])
    valid_layers = [
        int(row["layer"])
        for row in rows
        if minimum_flip <= float(row["future_flip_rate"]) <= maximum_flip
    ]
    passed = (
        final_accuracy >= float(thresholds["minimum_final_accuracy"])
        and len(valid_layers) >= int(thresholds["minimum_valid_layers_per_seed"])
    )
    return {
        "seed": int(rows[0]["seed"]),
        "final_accuracy": final_accuracy,
        "valid_future_flip_layers": valid_layers,
        "valid_layer_count": len(valid_layers),
        "passed": bool(passed),
    }


def save_selected_evaluation(
    model: Any,
    probe_data: dict[str, np.ndarray],
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trajectory = extract_trajectory(
        model,
        probe_data["tokens"],
        probe_data["labels"],
        batch_size=int(config["training"]["validation_batch_size"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"t1_u2_trajectory_seed_{seed}.npz", **trajectory)
    rows = trajectory_statistics(trajectory, seed)
    validity = seed_validity(rows, config["validity"])
    return rows, validity


def plot_screening(history: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for candidate_id, group in history.groupby("candidate_id"):
        axis.plot(
            group["step"],
            group["validation_final_accuracy"],
            marker="o",
            markersize=3,
            label=candidate_id,
        )
    axis.axhline(0.85, color="#c44e52", linestyle="--", linewidth=1.2, label="advancement gate")
    axis.set(
        title="T1-U2 validation-only candidate screening",
        xlabel="Training update",
        ylabel="Final-step validation accuracy",
        ylim=(0.45, 1.01),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    study = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    selection_seed = int(study["selection"]["seed"])
    tail_checkpoints = int(study["selection"]["tail_checkpoints"])
    candidate_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    trained: dict[str, tuple[Any, dict[str, np.ndarray], dict[str, Any]]] = {}

    for candidate in study["candidates"]:
        candidate_id = str(candidate["id"])
        print(f"SCREEN candidate={candidate_id} seed={selection_seed}", flush=True)
        config = candidate_config(study, candidate, selection_seed)
        candidate_output = candidates_dir / candidate_id
        model, history, probe_data = train_one_model(
            config,
            selection_seed,
            candidate_output / "checkpoints",
        )
        for row in history:
            history_rows.append({"candidate_id": candidate_id, **row})
        record = validation_record(candidate, model, history, tail_checkpoints)
        candidate_rows.append(record)
        trained[candidate_id] = (model, probe_data, config)
        pd.DataFrame(history).to_csv(candidate_output / "training_history.csv", index=False)
        print(
            f"SCREEN_RESULT candidate={candidate_id} "
            f"tail_mean={record['tail_mean_validation_accuracy']:.4f} "
            f"terminal={record['terminal_validation_accuracy']:.4f}",
            flush=True,
        )

    candidate_frame = pd.DataFrame(candidate_rows).sort_values(
        [
            "tail_mean_validation_accuracy",
            "terminal_validation_accuracy",
            "unique_parameters",
            "candidate_id",
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    candidate_frame.to_csv(output_dir / "candidate_results.csv", index=False)
    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(output_dir / "screening_training_history.csv", index=False)
    plot_screening(history_frame, output_dir / "t1_u2_candidate_screening.png")

    selected_row = candidate_frame.iloc[0].to_dict()
    selected_id = str(selected_row["candidate_id"])
    advancement_threshold = float(study["selection"]["minimum_validation_score"])
    advanced = float(selected_row["tail_mean_validation_accuracy"]) >= advancement_threshold
    selection_decision = {
        "selected_candidate": selected_id,
        "selected_record": selected_row,
        "minimum_validation_score": advancement_threshold,
        "advanced_to_independent_evaluation": bool(advanced),
        "ranking_rule": study["selection"]["ranking"],
        "sealed_data_rule": study["selection"]["sealed_data_rule"],
    }
    (output_dir / "selection_decision.json").write_text(
        json.dumps(selection_decision, indent=2), encoding="utf-8"
    )

    independent_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    replication_history: list[dict[str, Any]] = []
    if advanced:
        selected_model, selected_probe, selected_config = trained[selected_id]
        selected_dir = output_dir / "selected"
        selected_dir.mkdir(parents=True, exist_ok=True)
        with (selected_dir / "selected_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(selected_config, handle, sort_keys=False)
        rows, validity = save_selected_evaluation(
            selected_model,
            selected_probe,
            selected_config,
            selection_seed,
            selected_dir / "independent",
        )
        independent_rows.extend(rows)
        validity_rows.append(validity)

        replication_seed = int(study["selection"]["replication_seed"])
        selected_candidate = next(
            candidate for candidate in study["candidates"] if candidate["id"] == selected_id
        )
        replication_config = candidate_config(study, selected_candidate, replication_seed)
        print(f"REPLICATION candidate={selected_id} seed={replication_seed}", flush=True)
        replication_model, replication_history, replication_probe = train_one_model(
            replication_config,
            replication_seed,
            selected_dir / "replication" / "checkpoints",
        )
        rows, validity = save_selected_evaluation(
            replication_model,
            replication_probe,
            replication_config,
            replication_seed,
            selected_dir / "replication" / "independent",
        )
        independent_rows.extend(rows)
        validity_rows.append(validity)
        pd.DataFrame(replication_history).to_csv(
            selected_dir / "replication" / "training_history.csv", index=False
        )

    if independent_rows:
        pd.DataFrame(independent_rows).to_csv(
            output_dir / "independent_trajectory_statistics.csv", index=False
        )
    if validity_rows:
        pd.DataFrame(validity_rows).to_csv(output_dir / "independent_validity.csv", index=False)

    all_required_passed = bool(validity_rows) and all(row["passed"] for row in validity_rows)
    if not advanced:
        status = "screen_failed_advancement_gate"
    elif all_required_passed:
        status = "passed_ready_to_freeze_for_confirmatory_experiment"
    else:
        status = "failed_independent_learning_or_target_validity"

    summary = {
        "experiment": study["experiment"],
        "status": status,
        "historical_reference": study["historical_reference"],
        "selection": selection_decision,
        "candidate_results": candidate_frame.to_dict(orient="records"),
        "independent_validity": {
            "thresholds": study["independent_validity"],
            "seeds": validity_rows,
            "all_required_passed": all_required_passed,
        },
        "probe_policy": study["probes"]["policy"],
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
