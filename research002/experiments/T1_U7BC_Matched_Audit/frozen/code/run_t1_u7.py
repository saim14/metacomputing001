#!/usr/bin/env python3
"""Run T1-U7: bounded two-hop learning and causal-dependence gate.

The exposed development seeds are trained and evaluated first. Fresh model
seeds are trained only if every development seed passes the full preregistered
gate. This runner deliberately contains no trajectory-probe fitting path.
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

from src.t1_experiment import (
    extract_trajectory,
    generate_task,
    generate_two_hop_counterfactual_suite,
    train_one_model,
    write_environment_manifest,
)
from src.tiny_transformer import numpy_softmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T1-U7 two-hop validity gate")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/t1_u7_two_hop_gate.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/t1_u7"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_protocol(study: dict[str, Any]) -> None:
    data = study["data"]
    if str(data["task_type"]) != "chain_lookup" or int(data["chain_hops"]) != 2:
        raise ValueError("T1-U7 requires a two-hop chain_lookup task")
    if int(data["pair_count"]) != 5:
        raise ValueError("T1-U7 causal controls require exactly five mappings")
    if int(data["key_count"]) <= int(data["pair_count"]):
        raise ValueError("key_count must leave an unused key for edge corruption")

    development = [int(seed) for seed in study["development_seeds"]]
    fresh = [int(seed) for seed in study["fresh_confirmation_seeds"]]
    if not development or not fresh:
        raise ValueError("development and fresh-confirmation seeds must be nonempty")
    if set(development) & set(fresh):
        raise ValueError("development and fresh-confirmation seeds must be disjoint")
    if development != [int(seed) for seed in study["scientific_boundary"]["development_seeds"]]:
        raise ValueError("scientific boundary must name the exact development seeds")

    phases = study["training"].get("curriculum", [])
    if not phases:
        raise ValueError("the tapered T1-U4-D training schedule must be explicit")
    if any(int(phase["steps"]) <= 0 for phase in phases):
        raise ValueError("every training phase must have a positive update count")
    if any(int(phase.get("data_overrides", {}).get("chain_hops", 2)) != 2 for phase in phases):
        raise ValueError("T1-U7 must train only on two-hop examples")
    if any("data_stream_id" not in phase for phase in phases):
        raise ValueError("every phase must pin its data stream")

    gate = study["development_gate"]
    chance_min = float(gate["corruption_accuracy_minimum"])
    chance_max = float(gate["corruption_accuracy_maximum"])
    if not (0.0 <= chance_min <= 0.5 <= chance_max <= 1.0):
        raise ValueError("the corruption-accuracy band must contain chance")
    if int(gate["minimum_valid_value_steps"]) > int(study["model"]["n_layers"]) - 1:
        raise ValueError("value-step gate exceeds available intermediate steps")
    if "No trajectory probe" not in str(study["probes"]["policy"]):
        raise ValueError("T1-U7 must explicitly prohibit trajectory probes")


def training_config(study: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "experiment": copy.deepcopy(study["experiment"]),
        "hypotheses": copy.deepcopy(study["hypotheses"]),
        "seeds": [int(seed)],
        "data": copy.deepcopy(study["data"]),
        "model": copy.deepcopy(study["model"]),
        "training": copy.deepcopy(study["training"]),
        "probes": {"policy": study["probes"]["policy"]},
    }


def predict_final(
    model: Any,
    tokens: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    for start in range(0, len(tokens), batch_size):
        stop = min(len(tokens), start + batch_size)
        _, logits, _ = model(tokens[start:stop])
        probabilities.append(numpy_softmax(logits[-1].data))
    return np.concatenate(probabilities, axis=0)


def trajectory_value_statistics(
    trajectory: dict[str, np.ndarray],
    *,
    seed: int,
    phase: str,
    positive_epsilon: float,
) -> list[dict[str, Any]]:
    probabilities = trajectory["probabilities"]
    labels = trajectory["labels"]
    predictions = np.argmax(probabilities, axis=-1)
    sample_index = np.arange(len(labels))
    final_step = probabilities.shape[1] - 1
    final_nll = -np.log(
        np.clip(probabilities[sample_index, final_step, labels], 1e-8, 1.0)
    )
    final_prediction = predictions[:, final_step]
    rows: list[dict[str, Any]] = []
    for step in range(1, final_step):
        current_nll = -np.log(
            np.clip(probabilities[sample_index, step, labels], 1e-8, 1.0)
        )
        value = current_nll - final_nll
        rows.append(
            {
                "evaluation_phase": phase,
                "seed": int(seed),
                "step": int(step),
                "samples": int(len(labels)),
                "current_accuracy": float(np.mean(predictions[:, step] == labels)),
                "final_accuracy": float(np.mean(final_prediction == labels)),
                "future_flip_rate": float(np.mean(predictions[:, step] != final_prediction)),
                "mean_current_nll": float(np.mean(current_nll)),
                "mean_final_nll": float(np.mean(final_nll)),
                "mean_remaining_nll_reduction": float(np.mean(value)),
                "std_remaining_nll_reduction": float(np.std(value, ddof=1)),
                "median_remaining_nll_reduction": float(np.median(value)),
                "iqr_remaining_nll_reduction": float(
                    np.quantile(value, 0.75) - np.quantile(value, 0.25)
                ),
                "positive_value_fraction": float(np.mean(value > positive_epsilon)),
                "negative_value_fraction": float(np.mean(value < -positive_epsilon)),
            }
        )
    return rows


def causal_metrics(
    model: Any,
    *,
    study: dict[str, Any],
    seed: int,
    phase: str,
    causal_offset: int,
) -> dict[str, Any]:
    data = study["data"]
    suite = generate_two_hop_counterfactual_suite(
        int(data["causal_pairs"]),
        key_count=int(data["key_count"]),
        pair_count=int(data["pair_count"]),
        rng=np.random.default_rng(seed * 100 + causal_offset),
    )
    batch_size = int(study["training"]["validation_batch_size"])
    clean_probability = predict_final(
        model, suite["clean_tokens"], batch_size=batch_size
    )
    counterfactual_probability = predict_final(
        model, suite["counterfactual_tokens"], batch_size=batch_size
    )
    first_probability = predict_final(
        model, suite["first_edge_corrupted_tokens"], batch_size=batch_size
    )
    second_probability = predict_final(
        model, suite["second_edge_corrupted_tokens"], batch_size=batch_size
    )
    clean_prediction = np.argmax(clean_probability, axis=1)
    counterfactual_prediction = np.argmax(counterfactual_probability, axis=1)
    first_prediction = np.argmax(first_probability, axis=1)
    second_prediction = np.argmax(second_probability, axis=1)
    labels = suite["clean_labels"]
    counterfactual_labels = suite["counterfactual_labels"]
    return {
        "evaluation_phase": phase,
        "seed": int(seed),
        "pairs": int(len(labels)),
        "causal_clean_accuracy": float(np.mean(clean_prediction == labels)),
        "counterfactual_accuracy": float(
            np.mean(counterfactual_prediction == counterfactual_labels)
        ),
        "counterfactual_pair_accuracy": float(
            np.mean(
                (clean_prediction == labels)
                & (counterfactual_prediction == counterfactual_labels)
            )
        ),
        "counterfactual_prediction_flip_consistency": float(
            np.mean(counterfactual_prediction == 1 - clean_prediction)
        ),
        "first_edge_corrupted_accuracy": float(np.mean(first_prediction == labels)),
        "second_edge_corrupted_accuracy": float(np.mean(second_prediction == labels)),
        "second_edge_oracle_accuracy": float(
            np.mean(suite["second_edge_random_answers"] == labels)
        ),
    }


def evaluate_seed_gate(
    trajectory_rows: list[dict[str, Any]],
    causal: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    final_accuracy = float(trajectory_rows[0]["final_accuracy"])
    value_steps = [
        int(row["step"])
        for row in trajectory_rows
        if float(row["mean_remaining_nll_reduction"])
        >= float(thresholds["minimum_mean_remaining_nll_reduction"])
        and float(row["std_remaining_nll_reduction"])
        >= float(thresholds["minimum_std_remaining_nll_reduction"])
    ]
    accuracy_passed = final_accuracy >= float(thresholds["minimum_final_accuracy"])
    counterfactual_passed = float(causal["counterfactual_pair_accuracy"]) >= float(
        thresholds["minimum_counterfactual_pair_accuracy"]
    )
    chance_min = float(thresholds["corruption_accuracy_minimum"])
    chance_max = float(thresholds["corruption_accuracy_maximum"])
    first_edge_passed = chance_min <= float(
        causal["first_edge_corrupted_accuracy"]
    ) <= chance_max
    second_edge_passed = chance_min <= float(
        causal["second_edge_corrupted_accuracy"]
    ) <= chance_max
    value_passed = len(value_steps) >= int(thresholds["minimum_valid_value_steps"])
    passed = (
        accuracy_passed
        and counterfactual_passed
        and first_edge_passed
        and second_edge_passed
        and value_passed
    )
    return {
        "evaluation_phase": str(trajectory_rows[0]["evaluation_phase"]),
        "seed": int(trajectory_rows[0]["seed"]),
        "final_accuracy": final_accuracy,
        "counterfactual_pair_accuracy": float(causal["counterfactual_pair_accuracy"]),
        "first_edge_corrupted_accuracy": float(causal["first_edge_corrupted_accuracy"]),
        "second_edge_corrupted_accuracy": float(causal["second_edge_corrupted_accuracy"]),
        "valid_value_steps": value_steps,
        "valid_value_step_count": int(len(value_steps)),
        "accuracy_gate_passed": bool(accuracy_passed),
        "counterfactual_gate_passed": bool(counterfactual_passed),
        "first_edge_gate_passed": bool(first_edge_passed),
        "second_edge_gate_passed": bool(second_edge_passed),
        "value_gate_passed": bool(value_passed),
        "passed": bool(passed),
    }


def generate_clean_evaluation(
    study: dict[str, Any], seed: int, sample_offset: int
) -> dict[str, np.ndarray]:
    data = study["data"]
    tokens, labels = generate_task(
        int(data["probe_samples"]),
        data_config=data,
        rng=np.random.default_rng(seed * 100 + sample_offset),
    )
    return {"tokens": tokens, "labels": labels}


def run_seed_set(
    study: dict[str, Any],
    *,
    seeds: list[int],
    phase: str,
    sample_offset: int,
    causal_offset: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    histories: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    thresholds = study["development_gate"]

    for seed in seeds:
        print(f"{phase.upper()}_TRAIN seed={seed}", flush=True)
        seed_dir = output_dir / phase / f"seed_{seed}"
        config = training_config(study, seed)
        model, history, generated_development_data = train_one_model(
            config,
            seed,
            seed_dir / "checkpoints",
        )
        for row in history:
            histories.append({"evaluation_phase": phase, **row})

        if phase == "development" and sample_offset == 37:
            clean_data = generated_development_data
        else:
            clean_data = generate_clean_evaluation(study, seed, sample_offset)
        trajectory = extract_trajectory(
            model,
            clean_data["tokens"],
            clean_data["labels"],
            batch_size=int(study["training"]["validation_batch_size"]),
        )
        trajectory_dir = seed_dir / "trajectory"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trajectory_dir / f"t1_u7_{phase}_trajectory_seed_{seed}.npz",
            **trajectory,
        )
        seed_trajectory_rows = trajectory_value_statistics(
            trajectory,
            seed=seed,
            phase=phase,
            positive_epsilon=float(thresholds["value_positive_epsilon"]),
        )
        seed_causal = causal_metrics(
            model,
            study=study,
            seed=seed,
            phase=phase,
            causal_offset=causal_offset,
        )
        seed_gate = evaluate_seed_gate(seed_trajectory_rows, seed_causal, thresholds)
        trajectory_rows.extend(seed_trajectory_rows)
        causal_rows.append(seed_causal)
        gate_rows.append(seed_gate)
        print(
            f"{phase.upper()}_RESULT seed={seed} "
            f"accuracy={seed_gate['final_accuracy']:.5f} "
            f"counterfactual_pair={seed_gate['counterfactual_pair_accuracy']:.5f} "
            f"edge1={seed_gate['first_edge_corrupted_accuracy']:.5f} "
            f"edge2={seed_gate['second_edge_corrupted_accuracy']:.5f} "
            f"value_steps={seed_gate['valid_value_step_count']} "
            f"passed={seed_gate['passed']}",
            flush=True,
        )

    return (
        pd.DataFrame(histories),
        pd.DataFrame(trajectory_rows),
        pd.DataFrame(causal_rows),
        pd.DataFrame(gate_rows),
    )


def plot_results(
    training: pd.DataFrame,
    trajectory: pd.DataFrame,
    causal: pd.DataFrame,
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    phases = {"development": "-", "fresh_confirmation": "--"}
    for (phase, seed), group in training.groupby(["evaluation_phase", "seed"]):
        axes[0, 0].plot(
            group["step"],
            group["validation_final_accuracy"],
            linestyle=phases.get(str(phase), "-"),
            label=f"{phase}: {seed}",
        )
    axes[0, 0].axhline(
        float(gate["minimum_final_accuracy"]), color="#c44e52", linestyle=":"
    )
    axes[0, 0].set(
        title="Two-hop learning",
        xlabel="Training update",
        ylabel="Validation accuracy",
        ylim=(0.45, 1.01),
    )
    axes[0, 0].legend(frameon=False, fontsize=7)

    for (phase, seed), group in trajectory.groupby(["evaluation_phase", "seed"]):
        axes[0, 1].plot(
            group["step"],
            group["current_accuracy"],
            marker="o",
            linestyle=phases.get(str(phase), "-"),
            label=f"{phase}: {seed}",
        )
        axes[1, 0].plot(
            group["step"],
            group["mean_remaining_nll_reduction"],
            marker="o",
            linestyle=phases.get(str(phase), "-"),
            label=f"{phase}: {seed}",
        )
    axes[0, 1].set(
        title="Accuracy across refinement steps",
        xlabel="Refinement step",
        ylabel="Accuracy",
        ylim=(0.0, 1.01),
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

    metric_names = [
        "causal_clean_accuracy",
        "counterfactual_accuracy",
        "first_edge_corrupted_accuracy",
        "second_edge_corrupted_accuracy",
    ]
    labels = ["Clean", "Counterfactual", "Break first", "Break second"]
    x = np.arange(len(metric_names))
    development = causal[causal["evaluation_phase"] == "development"]
    means = [float(development[name].mean()) for name in metric_names]
    axes[1, 1].bar(x, means, color=["#4c78a8", "#59a14f", "#bab0ab", "#9c755f"])
    for index, name in enumerate(metric_names):
        values = development[name].to_numpy(dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0.0]
        axes[1, 1].scatter(index + np.asarray(jitter), values, color="#222222", s=18)
    axes[1, 1].axhspan(
        float(gate["corruption_accuracy_minimum"]),
        float(gate["corruption_accuracy_maximum"]),
        color="#dddddd",
        alpha=0.5,
    )
    axes[1, 1].set(
        title="Causal chain controls (development)",
        ylabel="Final accuracy",
        xticks=x,
        xticklabels=labels,
        ylim=(0.0, 1.01),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    fig.suptitle("T1-U7 bounded two-hop validity gate", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records") if not frame.empty else []


def run_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    study = load_config(config_path)
    validate_protocol(study)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / "preregistered_protocol.yaml")
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
    dev_history.to_csv(output_dir / "development_training_history.csv", index=False)
    dev_trajectory.to_csv(
        output_dir / "development_trajectory_value_statistics.csv", index=False
    )
    dev_causal.to_csv(output_dir / "development_causal_metrics.csv", index=False)
    dev_gate.to_csv(output_dir / "development_seed_gate.csv", index=False)
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
        study["development_gate"],
        output_dir / "t1_u7_two_hop_gate.png",
    )

    if str(study["experiment"]["id"]).endswith("-smoke"):
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
        "advance_to_t1_u8": bool(development_passed and fresh_passed),
    }
    (output_dir / "advancement_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment": study["experiment"],
        "status": status,
        "hypotheses": study["hypotheses"],
        "scientific_boundary": study["scientific_boundary"],
        "configuration": {
            "development_seeds": development_seeds,
            "fresh_confirmation_seeds": study["fresh_confirmation_seeds"],
            "data": study["data"],
            "model": study["model"],
            "training": study["training"],
        },
        "gate_thresholds": study["development_gate"],
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
