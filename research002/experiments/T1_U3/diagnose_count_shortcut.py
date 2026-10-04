#!/usr/bin/env python3
"""Quantify label information in value-token counts for T1 task generators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.t1_experiment import generate_chain_lookup_task, generate_lookup_task


def count_features(tokens: np.ndarray, key_count: int) -> np.ndarray:
    zero = np.sum(tokens == key_count, axis=1)
    one = np.sum(tokens == key_count + 1, axis=1)
    return np.stack([zero, one, one - zero], axis=1)


def evaluate(
    train_tokens: np.ndarray,
    train_labels: np.ndarray,
    test_tokens: np.ndarray,
    test_labels: np.ndarray,
    key_count: int,
) -> dict[str, float]:
    probe = LogisticRegression(max_iter=1000)
    probe.fit(count_features(train_tokens, key_count), train_labels)
    probability = probe.predict_proba(count_features(test_tokens, key_count))[:, 1]
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(test_labels, prediction)),
        "roc_auc": float(roc_auc_score(test_labels, probability)),
        "mean_value_zero_count": float(np.mean(np.sum(test_tokens == key_count, axis=1))),
        "mean_value_one_count": float(np.mean(np.sum(test_tokens == key_count + 1, axis=1))),
    }


def main(output: Path) -> None:
    key_count = 8
    train_size = 30000
    test_size = 30000
    original_train = generate_lookup_task(
        train_size,
        key_count=key_count,
        pair_count=4,
        rng=np.random.default_rng(501),
    )
    original_test = generate_lookup_task(
        test_size,
        key_count=key_count,
        pair_count=4,
        rng=np.random.default_rng(502),
    )
    balanced_train = generate_chain_lookup_task(
        train_size,
        key_count=key_count,
        pair_count=4,
        chain_hops=1,
        rng=np.random.default_rng(601),
    )
    balanced_test = generate_chain_lookup_task(
        test_size,
        key_count=key_count,
        pair_count=4,
        chain_hops=1,
        rng=np.random.default_rng(602),
    )
    result = {
        "diagnostic": "value-token-count-only logistic probe",
        "train_samples_per_condition": train_size,
        "test_samples_per_condition": test_size,
        "original_direct_lookup": evaluate(*original_train, *original_test, key_count),
        "balanced_one_hop": evaluate(*balanced_train, *balanced_test, key_count),
        "analytic_original_majority_accuracy_with_random_ties": 0.6875,
        "interpretation": (
            "The original generator exposes a count-based shortcut; the balanced generator "
            "fixes both value-token counts at two per sample and removes that signal."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/t1_1_diagnostics/count_shortcut.json"))
    arguments = parser.parse_args()
    main(arguments.output)
