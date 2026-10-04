"""Training, trajectory extraction, controls, probes, and reporting for T1."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .tiny_autograd import Adam, cross_entropy
from .tiny_transformer import ModelConfig, TinyTransformer, numpy_softmax


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate_lookup_task(
    n_samples: int,
    *,
    key_count: int,
    pair_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a two-hop key/value retrieval task.

    Layout: [CLS, query, key_1, value_1, ..., key_P, value_P].  The label is
    the binary value adjacent to the key that matches the query.
    """
    value_zero = key_count
    cls_token = key_count + 2
    sequence_length = 2 + 2 * pair_count
    tokens = np.empty((n_samples, sequence_length), dtype=np.int64)
    labels = rng.integers(0, 2, size=n_samples, dtype=np.int64)
    tokens[:, 0] = cls_token

    for sample in range(n_samples):
        keys = rng.choice(key_count, size=pair_count, replace=False)
        target_pair = int(rng.integers(pair_count))
        values = rng.integers(0, 2, size=pair_count, dtype=np.int64)
        values[target_pair] = labels[sample]
        tokens[sample, 1] = keys[target_pair]
        tokens[sample, 2::2] = keys
        tokens[sample, 3::2] = value_zero + values
    return tokens, labels


def generate_chain_lookup_task(
    n_samples: int,
    *,
    key_count: int,
    pair_count: int,
    chain_hops: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a multi-hop pointer-chasing task without bag-of-values leakage.

    A query key starts a chain of ``chain_hops`` mappings. The terminal mapping
    points to the binary label token. Distractor mappings contain enough value
    tokens to keep the total counts of value-zero and value-one equal in every
    sample, so the label cannot be recovered from token counts alone.
    """
    if chain_hops < 1:
        raise ValueError("chain_hops must be at least 1")
    if pair_count < chain_hops + 3:
        raise ValueError("pair_count must leave at least three distractor mappings")
    if key_count < pair_count:
        raise ValueError("key_count must be at least pair_count")

    value_zero = key_count
    cls_token = key_count + 2
    sequence_length = 2 + 2 * pair_count
    tokens = np.empty((n_samples, sequence_length), dtype=np.int64)
    labels = rng.integers(0, 2, size=n_samples, dtype=np.int64)
    tokens[:, 0] = cls_token

    for sample in range(n_samples):
        source_keys = rng.choice(key_count, size=pair_count, replace=False)
        chain_keys = source_keys[:chain_hops]
        sources = list(chain_keys)
        destinations: list[int] = list(chain_keys[1:]) + [value_zero + int(labels[sample])]

        distractor_sources = list(source_keys[chain_hops:])
        # Together with the terminal value, these three distractors produce
        # exactly two value-zero and two value-one tokens for either label.
        distractor_destinations = [
            value_zero + int(labels[sample]),
            value_zero + (1 - int(labels[sample])),
            value_zero + (1 - int(labels[sample])),
        ]
        remaining = len(distractor_sources) - len(distractor_destinations)
        if remaining > 0:
            distractor_destinations.extend(
                rng.integers(0, key_count, size=remaining, dtype=np.int64).tolist()
            )
        rng.shuffle(distractor_destinations)
        sources.extend(distractor_sources)
        destinations.extend(distractor_destinations)

        pair_order = rng.permutation(pair_count)
        source_array = np.asarray(sources, dtype=np.int64)[pair_order]
        destination_array = np.asarray(destinations, dtype=np.int64)[pair_order]
        tokens[sample, 1] = chain_keys[0]
        tokens[sample, 2::2] = source_array
        tokens[sample, 3::2] = destination_array
    return tokens, labels


def generate_task(
    n_samples: int,
    *,
    data_config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    task_type = data_config.get("task_type", "direct_lookup")
    common = {
        "n_samples": n_samples,
        "key_count": int(data_config["key_count"]),
        "pair_count": int(data_config["pair_count"]),
        "rng": rng,
    }
    if task_type == "direct_lookup":
        return generate_lookup_task(**common)
    if task_type == "chain_lookup":
        return generate_chain_lookup_task(
            **common,
            chain_hops=int(data_config["chain_hops"]),
        )
    raise ValueError(f"unknown task_type: {task_type}")


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def train_one_model(
    config: dict[str, Any],
    seed: int,
    checkpoint_dir: Path,
) -> tuple[TinyTransformer, list[dict[str, Any]], dict[str, np.ndarray]]:
    data_cfg = config["data"]
    model_cfg = ModelConfig(
        vocab_size=data_cfg["key_count"] + 3,
        sequence_length=2 + 2 * data_cfg["pair_count"],
        **config["model"],
    )
    rng = np.random.default_rng(seed)
    probe_tokens, probe_labels = generate_task(
        data_cfg["probe_samples"],
        data_config=data_cfg,
        rng=np.random.default_rng(seed * 100 + 37),
    )

    model = TinyTransformer(model_cfg, seed=seed)
    train_cfg = config["training"]
    optimizer = Adam(
        model.parameters(),
        learning_rate=float(train_cfg["learning_rate"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
    )
    history: list[dict[str, Any]] = []
    batch_size = int(train_cfg["batch_size"])
    log_every = int(train_cfg["log_every"])
    curriculum = train_cfg.get("curriculum")
    if curriculum:
        phases = curriculum
    else:
        phases = [
            {
                "name": "target",
                "steps": int(train_cfg["steps"]),
                "data_overrides": {},
                "auxiliary_loss_weight": float(train_cfg["auxiliary_loss_weight"]),
                "learning_rate": float(train_cfg["learning_rate"]),
            }
        ]
    total_steps = int(sum(int(phase["steps"]) for phase in phases))
    global_step = 0
    started = time.perf_counter()

    for phase_index, phase in enumerate(phases):
        phase_name = str(phase.get("name", f"phase_{phase_index + 1}"))
        phase_steps = int(phase["steps"])
        phase_data_cfg = {**data_cfg, **phase.get("data_overrides", {})}
        if curriculum:
            data_stream_id = int(phase.get("data_stream_id", phase_index))
            data_seed_base = seed * 10000 + data_stream_id * 100
        else:
            data_seed_base = seed * 100
        train_tokens, train_labels = generate_task(
            phase_data_cfg["train_samples"],
            data_config=phase_data_cfg,
            rng=np.random.default_rng(data_seed_base + 11),
        )
        val_tokens, val_labels = generate_task(
            phase_data_cfg["validation_samples"],
            data_config=phase_data_cfg,
            rng=np.random.default_rng(data_seed_base + 23),
        )
        auxiliary_weight = float(
            phase.get("auxiliary_loss_weight", train_cfg["auxiliary_loss_weight"])
        )
        optimizer.learning_rate = float(phase.get("learning_rate", train_cfg["learning_rate"]))

        for phase_step in range(1, phase_steps + 1):
            global_step += 1
            batch_index = rng.integers(0, len(train_tokens), size=batch_size)
            _, logits_by_layer, _ = model(train_tokens[batch_index])
            final_loss = cross_entropy(logits_by_layer[-1], train_labels[batch_index])
            earlier_losses = [
                cross_entropy(logits, train_labels[batch_index]) for logits in logits_by_layer[1:-1]
            ]
            if earlier_losses:
                auxiliary_loss = sum(earlier_losses[1:], earlier_losses[0]) / len(earlier_losses)
                loss = final_loss + auxiliary_weight * auxiliary_loss
            else:
                loss = final_loss
            optimizer.zero_grad()
            loss.backward()
            grad_norm = optimizer.step()

            if phase_step == 1 or phase_step % log_every == 0 or phase_step == phase_steps:
                validation_subset = min(len(val_tokens), int(train_cfg["validation_batch_size"]))
                _, validation_logits, _ = model(val_tokens[:validation_subset])
                layer_accuracies = [
                    accuracy_from_logits(layer_logits.data, val_labels[:validation_subset])
                    for layer_logits in validation_logits
                ]
                row = {
                    "seed": seed,
                    "step": global_step,
                    "phase": phase_name,
                    "phase_step": phase_step,
                    "phase_chain_hops": phase_data_cfg.get("chain_hops"),
                    "auxiliary_loss_weight": auxiliary_weight,
                    "learning_rate": optimizer.learning_rate,
                    "loss": float(loss.data),
                    "final_loss": float(final_loss.data),
                    "grad_norm": grad_norm,
                    "validation_final_accuracy": layer_accuracies[-1],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                row.update({f"validation_accuracy_layer_{i}": value for i, value in enumerate(layer_accuracies)})
                history.append(row)
                print(
                    f"seed={seed} phase={phase_name} phase_step={phase_step:4d}/{phase_steps} "
                    f"total={global_step:4d}/{total_steps} loss={row['loss']:.4f} "
                    f"val_acc={row['validation_final_accuracy']:.3f} grad={grad_norm:.3f}",
                    flush=True,
                )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_dir / f"t1_transformer_seed_{seed}.npz")
    probe_data = {"tokens": probe_tokens, "labels": probe_labels}
    return model, history, probe_data


def extract_trajectory(
    model: TinyTransformer,
    tokens: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    cls_states: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    attention_summaries: list[np.ndarray] = []

    for start in range(0, len(tokens), batch_size):
        stop = min(len(tokens), start + batch_size)
        states, logits, attentions = model(tokens[start:stop])
        cls_states.append(np.stack([state.data[:, 0, :] for state in states], axis=1))
        probabilities.append(np.stack([numpy_softmax(value.data) for value in logits], axis=1))

        per_layer = []
        for attention in attentions:
            cls_attention = np.clip(attention.data[:, :, 0, :], 1e-8, 1.0)
            entropy = -np.sum(cls_attention * np.log(cls_attention), axis=-1) / np.log(cls_attention.shape[-1])
            maximum = np.max(cls_attention, axis=-1)
            query_mass = cls_attention[:, :, 1]
            per_layer.append(np.stack([entropy, maximum, query_mass], axis=-1))
        attention_summaries.append(np.stack(per_layer, axis=1))

    state_array = np.concatenate(cls_states, axis=0)
    probability_array = np.concatenate(probabilities, axis=0)
    attention_array = np.concatenate(attention_summaries, axis=0)
    predictive_entropy = -np.sum(
        np.clip(probability_array, 1e-8, 1.0) * np.log2(np.clip(probability_array, 1e-8, 1.0)),
        axis=-1,
        keepdims=True,
    )
    probability_margin = np.abs(
        probability_array[:, :, 1:2] - probability_array[:, :, 0:1]
    )
    state_features = np.concatenate(
        [state_array, probability_array, predictive_entropy, probability_margin], axis=-1
    ).astype(np.float32)
    return {
        "tokens": tokens,
        "labels": labels,
        "cls_states": state_array.astype(np.float32),
        "probabilities": probability_array.astype(np.float32),
        "state_features": state_features,
        "attention_summaries": attention_array.astype(np.float32),
    }


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high if high < 1.0 else probability <= high)
        if np.any(mask):
            result += (np.sum(mask) / total) * abs(float(np.mean(y_true[mask])) - float(np.mean(probability[mask])))
    return float(result)


def build_control_features(
    features: np.ndarray,
    layer: int,
    *,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    history = features[:, : layer + 1, :]
    current = features[:, layer, :]
    slots = layer + 1
    scale = np.sqrt(float(slots))
    ordered = history.reshape(len(features), -1) / scale
    dimension_matched = np.repeat(current[:, None, :], slots, axis=1).reshape(len(features), -1) / scale

    shuffled = history.copy()
    if layer > 1:
        for sample in range(len(features)):
            shuffled[sample, :layer, :] = shuffled[sample, rng.permutation(layer), :]
    shuffled_history = shuffled.reshape(len(features), -1) / scale

    mismatched = history.copy()
    if layer > 0:
        donor_order = rng.permutation(len(features))
        mismatched[:, :layer, :] = history[donor_order, :layer, :]
    sample_mismatched = mismatched.reshape(len(features), -1) / scale
    return {
        "current": current,
        "ordered_history": ordered,
        "shuffled_history": shuffled_history,
        "dimension_matched": dimension_matched,
        "sample_mismatched_history": sample_mismatched,
    }


def build_attention_control_features(
    state_features: np.ndarray,
    attention_summaries: np.ndarray,
    layer: int,
    *,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Construct attention-history conditions while holding current state fixed.

    ``layer`` is one-indexed with respect to Transformer blocks. Attention index
    zero therefore belongs to layer one.
    """
    current_state = state_features[:, layer, :]
    history = attention_summaries[:, :layer, :, :].reshape(len(state_features), layer, -1)
    current_attention = history[:, -1, :]
    scale = np.sqrt(float(layer))

    current_plus_current = np.concatenate([current_state, current_attention], axis=1)
    ordered = np.concatenate(
        [current_state, history.reshape(len(state_features), -1) / scale], axis=1
    )
    dimension_matched_history = np.repeat(
        current_attention[:, None, :], layer, axis=1
    )
    dimension_matched = np.concatenate(
        [current_state, dimension_matched_history.reshape(len(state_features), -1) / scale],
        axis=1,
    )

    shuffled = history.copy()
    if layer > 2:
        for sample in range(len(state_features)):
            shuffled[sample, : layer - 1, :] = shuffled[
                sample, rng.permutation(layer - 1), :
            ]
    shuffled_matrix = np.concatenate(
        [current_state, shuffled.reshape(len(state_features), -1) / scale], axis=1
    )

    mismatched = history.copy()
    if layer > 1:
        donor_order = rng.permutation(len(state_features))
        mismatched[:, : layer - 1, :] = history[donor_order, : layer - 1, :]
    mismatched_matrix = np.concatenate(
        [current_state, mismatched.reshape(len(state_features), -1) / scale], axis=1
    )
    return {
        "current_plus_current_attention": current_plus_current,
        "ordered_attention_history": ordered,
        "shuffled_attention_history": shuffled_matrix,
        "attention_dimension_matched": dimension_matched,
        "sample_mismatched_attention_history": mismatched_matrix,
    }


def residual_history_features(
    features: np.ndarray,
    layer: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    *,
    seed: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    current = features[:, layer, :]
    past = features[:, :layer, :].reshape(len(features), -1)
    residualizer = Ridge(alpha=alpha)
    residualizer.fit(current[train_index], past[train_index])
    train_residual = past[train_index] - residualizer.predict(current[train_index])
    test_residual = past[test_index] - residualizer.predict(current[test_index])

    # Near-constant historical blocks (notably the initial CLS embedding) can
    # leave floating-point residuals around 1e-8.  A downstream StandardScaler
    # would magnify those numerical artifacts into spurious features.
    active = np.std(train_residual, axis=0) > 1e-6
    train_residual[:, ~active] = 0.0
    test_residual[:, ~active] = 0.0
    scale = np.sqrt(float(layer + 1))
    train_features = np.concatenate([train_residual, current[train_index]], axis=1) / scale
    test_features = np.concatenate([test_residual, current[test_index]], axis=1) / scale
    return train_features, test_features


def evaluate_binary_probe(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )
    probe.fit(x_train, y_train)
    probability = np.clip(probe.predict_proba(x_test)[:, 1], 1e-7, 1.0 - 1e-7)
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "pr_auc": float(average_precision_score(y_test, probability)),
        "log_loss": float(log_loss(y_test, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, probability)),
        "ece_10": expected_calibration_error(y_test, probability),
        "accuracy": float(accuracy_score(y_test, prediction)),
    }


def evaluate_regression_probe(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float]:
    probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    probe.fit(x_train, y_train)
    prediction = probe.predict(x_test)
    correlation = stats.spearmanr(y_test, prediction).statistic
    return {
        "r2": float(r2_score(y_test, prediction)),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "spearman": float(correlation),
    }


def probe_trajectory(
    trajectory: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features = trajectory["state_features"]
    probabilities = trajectory["probabilities"]
    attention_summaries = trajectory["attention_summaries"]
    labels = trajectory["labels"]
    predictions = np.argmax(probabilities, axis=-1)
    final_layer = probabilities.shape[1] - 1
    final_prediction = predictions[:, final_layer]
    probe_cfg = config["probes"]
    raw_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for layer in range(1, final_layer):
        current_prediction = predictions[:, layer]
        future_flip = (current_prediction != final_prediction).astype(np.int64)
        future_correction = (
            (current_prediction != labels) & (final_prediction == labels)
        ).astype(np.int64)
        current_nll = -np.log(np.clip(probabilities[np.arange(len(labels)), layer, labels], 1e-8, 1.0))
        final_nll = -np.log(np.clip(probabilities[np.arange(len(labels)), final_layer, labels], 1e-8, 1.0))
        delta_nll = current_nll - final_nll

        sample_rows.append(
            {
                "seed": seed,
                "layer": layer,
                "samples": len(labels),
                "current_accuracy": float(np.mean(current_prediction == labels)),
                "final_accuracy": float(np.mean(final_prediction == labels)),
                "future_flip_rate": float(np.mean(future_flip)),
                "future_correction_rate": float(np.mean(future_correction)),
                "mean_delta_nll": float(np.mean(delta_nll)),
                "median_delta_nll": float(np.median(delta_nll)),
            }
        )

        split_target = future_flip if min(np.sum(future_flip), np.sum(1 - future_flip)) >= 4 else None
        indices = np.arange(len(labels))
        train_index, test_index = train_test_split(
            indices,
            test_size=float(probe_cfg["test_fraction"]),
            random_state=seed * 100 + layer,
            stratify=split_target,
        )
        controls = build_control_features(
            features,
            layer,
            rng=np.random.default_rng(seed * 1000 + layer),
        )
        state_condition_names = set(controls)
        attention_controls = build_attention_control_features(
            features,
            attention_summaries,
            layer,
            rng=np.random.default_rng(seed * 2000 + layer),
        )
        controls.update(attention_controls)
        residual_train, residual_test = residual_history_features(
            features,
            layer,
            train_index,
            test_index,
            seed=seed * 100 + layer,
            alpha=float(probe_cfg["ridge_alpha"]),
        )

        binary_targets = {
            "future_flip": future_flip,
            "future_correction": future_correction,
        }
        for target_name, target in binary_targets.items():
            positives = int(np.sum(target))
            negatives = int(len(target) - positives)
            if min(positives, negatives) < int(probe_cfg["minimum_class_count"]):
                raw_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "task": target_name,
                        "task_kind": "binary",
                        "condition": "all",
                        "feature_family": "all",
                        "status": "skipped_insufficient_class_count",
                        "positive_rate": positives / len(target),
                    }
                )
                continue
            for condition, matrix in controls.items():
                metrics = evaluate_binary_probe(
                    matrix[train_index],
                    matrix[test_index],
                    target[train_index],
                    target[test_index],
                    seed=seed,
                )
                raw_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "task": target_name,
                        "task_kind": "binary",
                        "condition": condition,
                        "feature_family": (
                            "state_history" if condition in state_condition_names else "attention_history"
                        ),
                        "status": "ok",
                        "input_dimension": matrix.shape[1],
                        "positive_rate": positives / len(target),
                        **metrics,
                    }
                )
            metrics = evaluate_binary_probe(
                residual_train,
                residual_test,
                target[train_index],
                target[test_index],
                seed=seed,
            )
            raw_rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "task": target_name,
                    "task_kind": "binary",
                    "condition": "residual_history",
                    "feature_family": "state_history",
                    "status": "ok",
                    "input_dimension": residual_train.shape[1],
                    "positive_rate": positives / len(target),
                    **metrics,
                }
            )

        for condition, matrix in controls.items():
            metrics = evaluate_regression_probe(
                matrix[train_index],
                matrix[test_index],
                delta_nll[train_index],
                delta_nll[test_index],
                alpha=float(probe_cfg["ridge_alpha"]),
            )
            raw_rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "task": "delta_nll_to_final",
                    "task_kind": "regression",
                    "condition": condition,
                    "feature_family": (
                        "state_history" if condition in state_condition_names else "attention_history"
                    ),
                    "status": "ok",
                    "input_dimension": matrix.shape[1],
                    **metrics,
                }
            )
        metrics = evaluate_regression_probe(
            residual_train,
            residual_test,
            delta_nll[train_index],
            delta_nll[test_index],
            alpha=float(probe_cfg["ridge_alpha"]),
        )
        raw_rows.append(
            {
                "seed": seed,
                "layer": layer,
                "task": "delta_nll_to_final",
                "task_kind": "regression",
                "condition": "residual_history",
                "feature_family": "state_history",
                "status": "ok",
                "input_dimension": residual_train.shape[1],
                **metrics,
            }
        )
    return raw_rows, sample_rows


def compute_paired_effects(raw: pd.DataFrame) -> pd.DataFrame:
    comparison_families = {
        "state_history": {
            "ordered": "ordered_history",
            "comparators": [
                "current",
                "shuffled_history",
                "dimension_matched",
                "sample_mismatched_history",
            ],
        },
        "attention_history": {
            "ordered": "ordered_attention_history",
            "comparators": [
                "current_plus_current_attention",
                "shuffled_attention_history",
                "attention_dimension_matched",
                "sample_mismatched_attention_history",
            ],
        },
    }
    metric_directions = {
        "roc_auc": 1.0,
        "log_loss": -1.0,
        "r2": 1.0,
        "mae": -1.0,
        "spearman": 1.0,
    }
    rows: list[dict[str, Any]] = []
    ok = raw[raw["status"] == "ok"]
    for (seed, layer, task), group in ok.groupby(["seed", "layer", "task"]):
        for family, specification in comparison_families.items():
            ordered_condition = specification["ordered"]
            ordered = group[group["condition"] == ordered_condition]
            if ordered.empty:
                continue
            for comparator in specification["comparators"]:
                baseline = group[group["condition"] == comparator]
                if baseline.empty:
                    continue
                for metric, direction in metric_directions.items():
                    ordered_value = ordered.iloc[0].get(metric)
                    baseline_value = baseline.iloc[0].get(metric)
                    if pd.isna(ordered_value) or pd.isna(baseline_value):
                        continue
                    rows.append(
                        {
                            "seed": seed,
                            "layer": layer,
                            "task": task,
                            "feature_family": family,
                            "ordered_condition": ordered_condition,
                            "comparator": comparator,
                            "metric": metric,
                            "effect_favoring_ordered": direction * (ordered_value - baseline_value),
                        }
                    )
    return pd.DataFrame(rows)


def confidence_interval(values: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    if len(values) < 2:
        return None, None
    if float(np.std(values, ddof=1)) < 1e-15:
        mean = float(np.mean(values))
        return mean, mean
    sem = stats.sem(values)
    interval = stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=sem)
    return float(interval[0]), float(interval[1])


def evaluate_run_validity(
    sample_stats: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    validity = config.get("validity")
    if not validity:
        return {"status": "not_preregistered"}
    minimum_accuracy = float(validity["minimum_final_accuracy"])
    minimum_flip = float(validity["minimum_future_flip_rate"])
    maximum_flip = float(validity["maximum_future_flip_rate"])
    minimum_layers = int(validity["minimum_valid_layers_per_seed"])
    minimum_seed_fraction = float(validity["minimum_passing_seed_fraction"])
    seed_rows: list[dict[str, Any]] = []
    for seed, group in sample_stats.groupby("seed"):
        final_accuracy = float(group["final_accuracy"].iloc[0])
        valid_layer_mask = group["future_flip_rate"].between(minimum_flip, maximum_flip)
        valid_layers = group.loc[valid_layer_mask, "layer"].astype(int).tolist()
        passed = final_accuracy >= minimum_accuracy and len(valid_layers) >= minimum_layers
        seed_rows.append(
            {
                "seed": int(seed),
                "final_accuracy": final_accuracy,
                "valid_future_flip_layers": valid_layers,
                "valid_layer_count": len(valid_layers),
                "passed": bool(passed),
            }
        )
    passing_fraction = float(np.mean([row["passed"] for row in seed_rows]))
    return {
        "status": "passed" if passing_fraction >= minimum_seed_fraction else "failed",
        "thresholds": validity,
        "passing_seed_fraction": passing_fraction,
        "seeds": seed_rows,
    }


def create_summary(
    raw: pd.DataFrame,
    paired: pd.DataFrame,
    sample_stats: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    metrics_summary: list[dict[str, Any]] = []
    ok = raw[raw["status"] == "ok"]
    metric_names = ["roc_auc", "pr_auc", "log_loss", "brier", "ece_10", "r2", "mae", "spearman"]
    for keys, group in ok.groupby(["task", "condition"]):
        for metric in metric_names:
            values = group[metric].dropna().to_numpy(dtype=float) if metric in group else np.array([])
            if not len(values):
                continue
            low, high = confidence_interval(values)
            metrics_summary.append(
                {
                    "task": keys[0],
                    "condition": keys[1],
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

    effect_summary: list[dict[str, Any]] = []
    if not paired.empty:
        for keys, group in paired.groupby(["task", "feature_family", "comparator", "metric"]):
            values = group["effect_favoring_ordered"].to_numpy(dtype=float)
            low, high = confidence_interval(values)
            effect_summary.append(
                {
                    "task": keys[0],
                    "feature_family": keys[1],
                    "comparator": keys[2],
                    "metric": keys[3],
                    "n": int(len(values)),
                    "mean_effect_favoring_ordered": float(np.mean(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_fraction": float(np.mean(values > 0)),
                }
            )

    return {
        "experiment": config["experiment"],
        "hypotheses": config["hypotheses"],
        "configuration": {
            "seeds": config["seeds"],
            "data": config["data"],
            "model": config["model"],
            "training": config["training"],
            "probes": config["probes"],
            "validity": config.get("validity"),
        },
        "interpretation_boundary": (
            "Probe improvements establish incremental decodability under the stated controls; "
            "they do not by themselves establish causal use or metacognition."
        ),
        "training_and_target_statistics": sample_stats.to_dict(orient="records"),
        "run_validity": evaluate_run_validity(sample_stats, config),
        "metric_summary": metrics_summary,
        "paired_effect_summary": effect_summary,
    }


def plot_results(
    paired: pd.DataFrame,
    sample_stats: pd.DataFrame,
    output_path: Path,
) -> None:
    comparators = ["current", "dimension_matched", "shuffled_history", "sample_mismatched_history"]
    short_labels = ["Current", "Dimension\nmatched", "Shuffled\nhistory", "Sample\nmismatched"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)

    for seed, group in sample_stats.groupby("seed"):
        axes[0].plot(group["layer"], group["future_flip_rate"], marker="o", label=f"seed {seed}")
    axes[0].set(
        title="Target availability",
        xlabel="Current layer",
        ylabel="Future-flip rate",
        xticks=sorted(sample_stats["layer"].unique()),
        ylim=(-0.02, max(0.4, float(sample_stats["future_flip_rate"].max()) + 0.04)),
    )
    axes[0].legend(frameon=False, fontsize=8)

    panels = [
        (axes[1], "future_flip", "roc_auc", "Future flip: Δ ROC AUC"),
        (axes[2], "delta_nll_to_final", "r2", "Future NLL gain: Δ R²"),
    ]
    for axis, task, metric, title in panels:
        subset = paired[(paired["task"] == task) & (paired["metric"] == metric)]
        for index, comparator in enumerate(comparators):
            values = subset[subset["comparator"] == comparator]["effect_favoring_ordered"].to_numpy()
            if not len(values):
                continue
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            axis.scatter(index + jitter, values, color="#708090", alpha=0.65, s=24)
            low, high = confidence_interval(values)
            mean = float(np.mean(values))
            if low is not None:
                axis.vlines(index, low, high, color="#c44e52", linewidth=2.2)
            axis.scatter(index, mean, color="#c44e52", marker="D", s=44, zorder=3)
        axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        axis.set(title=title, ylabel="Effect favoring ordered history")
        axis.set_xticks(range(len(comparators)), short_labels, fontsize=8)

    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("T1 baseline: target saturation and paired control effects", fontsize=13)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_attention_results(paired: pd.DataFrame, output_path: Path) -> None:
    attention = paired[paired.get("feature_family") == "attention_history"]
    if attention.empty:
        return
    comparators = [
        "current_plus_current_attention",
        "attention_dimension_matched",
        "shuffled_attention_history",
        "sample_mismatched_attention_history",
    ]
    short_labels = ["Current\nattention", "Dimension\nmatched", "Shuffled\nhistory", "Sample\nmismatched"]
    panels = [
        ("future_flip", "roc_auc", "Future flip: Δ ROC AUC"),
        ("delta_nll_to_final", "r2", "Future NLL gain: Δ R²"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for axis, (task, metric, title) in zip(axes, panels):
        subset = attention[(attention["task"] == task) & (attention["metric"] == metric)]
        for index, comparator in enumerate(comparators):
            values = subset[subset["comparator"] == comparator]["effect_favoring_ordered"].to_numpy()
            if not len(values):
                continue
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            axis.scatter(index + jitter, values, color="#708090", alpha=0.65, s=24)
            low, high = confidence_interval(values)
            mean = float(np.mean(values))
            if low is not None:
                axis.vlines(index, low, high, color="#4c72b0", linewidth=2.2)
            axis.scatter(index, mean, color="#4c72b0", marker="D", s=44, zorder=3)
        axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        axis.set(title=title, ylabel="Effect favoring ordered attention history")
        axis.set_xticks(range(len(comparators)), short_labels, fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("T1.1 attention-history paired control effects", fontsize=13)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_environment_manifest(path: Path) -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "pyyaml": yaml.__version__,
        "deep_learning_framework": "none; local NumPy autodiff implementation",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiment(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    all_history: list[dict[str, Any]] = []
    all_raw: list[dict[str, Any]] = []
    all_sample_stats: list[dict[str, Any]] = []
    started = time.perf_counter()

    for seed in config["seeds"]:
        model, history, probe_data = train_one_model(config, int(seed), checkpoint_dir)
        all_history.extend(history)
        trajectory = extract_trajectory(
            model,
            probe_data["tokens"],
            probe_data["labels"],
            batch_size=int(config["training"]["validation_batch_size"]),
        )
        np.savez_compressed(trajectory_dir / f"t1_trajectory_seed_{seed}.npz", **trajectory)
        raw_rows, sample_rows = probe_trajectory(trajectory, config, seed=int(seed))
        all_raw.extend(raw_rows)
        all_sample_stats.extend(sample_rows)

    training = pd.DataFrame(all_history)
    raw = pd.DataFrame(all_raw)
    sample_stats = pd.DataFrame(all_sample_stats)
    paired = compute_paired_effects(raw)
    training.to_csv(output_dir / "training_history.csv", index=False)
    raw.to_csv(output_dir / "probe_metrics_raw.csv", index=False)
    sample_stats.to_csv(output_dir / "layer_target_statistics.csv", index=False)
    paired.to_csv(output_dir / "paired_effects.csv", index=False)
    summary = create_summary(raw, paired, sample_stats, config)
    summary["elapsed_seconds"] = time.perf_counter() - started
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_results(paired, sample_stats, output_dir / "t1_probe_comparison.png")
    plot_attention_results(paired, output_dir / "t1_attention_probe_comparison.png")
    write_environment_manifest(output_dir / "environment.json")
    return summary
