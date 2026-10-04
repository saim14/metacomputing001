"""T1-U7BC-M1: inference-only audit of frozen checkpoints and original suites."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "frozen/code"))
from run_t1_u7 import (evaluate_seed_gate, generate_clean_evaluation, load_config,
                       predict_final, trajectory_value_statistics)
from src.t1_experiment import extract_trajectory, generate_two_hop_counterfactual_suite
from src.tiny_transformer import ModelConfig, TinyTransformer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(arr):
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_model(study, version, seed):
    model = TinyTransformer(ModelConfig(
        vocab_size=study["data"]["key_count"] + 3,
        sequence_length=2 + 2 * study["data"]["pair_count"], **study["model"]), seed=seed)
    path = ROOT / "frozen" / version / f"seed_{seed}" / "checkpoint.npz"
    with np.load(path, allow_pickle=False) as saved:
        params = {p.name: p for p in model.parameters()}
        assert set(saved.files) == set(params)
        for name, p in params.items():
            assert p.data.shape == saved[name].shape
            p.data[...] = saved[name]
            # Same forward arithmetic; no differentiation graph is needed.
            p.requires_grad = False
            p.grad = None
    return model


def native_tables(version, seed):
    prefix = "fresh" if version == "U7B" and seed >= 811 else "development"
    base = ROOT / "frozen" / version
    return {name: pd.read_csv(base / f"{prefix}_{name}.csv").query("seed == @seed")
            for name in ["seed_gate", "causal_metrics", "trajectory_value_statistics"]}


def verify_native(trajectory, causal, values, gate, version, seed):
    with np.load(ROOT / "frozen" / version / f"seed_{seed}" / "native_reference.npz",
                 allow_pickle=False) as ref:
        for key in ["tokens", "labels"]:
            np.testing.assert_array_equal(trajectory[key], ref[key])
        np.testing.assert_allclose(trajectory["probabilities"], ref["probabilities"],
                                   rtol=1e-5, atol=2e-6)
        max_error = float(np.max(np.abs(trajectory["probabilities"] - ref["probabilities"])))
    tables = native_tables(version, seed)
    for field, value in causal.items():
        if field != "evaluation_phase":
            np.testing.assert_allclose(value, tables["causal_metrics"].iloc[0][field],
                                       rtol=1e-7, atol=1e-8)
    original_gate = tables["seed_gate"].iloc[0]
    for field, value in gate.items():
        if field not in ["evaluation_phase", "valid_value_steps"]:
            np.testing.assert_allclose(value, original_gate[field], rtol=1e-7, atol=1e-8)
    assert gate["valid_value_steps"] == json.loads(original_gate["valid_value_steps"])
    original_values = tables["trajectory_value_statistics"].set_index("step")
    for row in values:
        for field, value in row.items():
            if field != "evaluation_phase":
                expected = row["step"] if field == "step" else original_values.loc[row["step"], field]
                np.testing.assert_allclose(value, expected, rtol=1e-6, atol=2e-7)
    return max_error


def evaluate(model, study, seed, clean, suite):
    tr = extract_trajectory(model, clean["tokens"], clean["labels"],
                            batch_size=study["training"]["validation_batch_size"])
    pred = {name: np.argmax(predict_final(model, suite[name], batch_size=600), axis=1)
            for name in ["clean_tokens", "counterfactual_tokens",
                         "first_edge_corrupted_tokens", "second_edge_corrupted_tokens"]}
    labels = suite["clean_labels"]
    clean_ok = pred["clean_tokens"] == labels
    cf_ok = pred["counterfactual_tokens"] == suite["counterfactual_labels"]
    pair_ok = clean_ok & cf_ok
    causal = {
        "evaluation_phase": "matched_audit", "seed": seed, "pairs": len(labels),
        "causal_clean_accuracy": float(clean_ok.mean()),
        "counterfactual_accuracy": float(cf_ok.mean()),
        "counterfactual_pair_accuracy": float(pair_ok.mean()),
        "counterfactual_prediction_flip_consistency": float(np.mean(
            pred["counterfactual_tokens"] == 1 - pred["clean_tokens"])),
        "first_edge_corrupted_accuracy": float(np.mean(pred["first_edge_corrupted_tokens"] == labels)),
        "second_edge_corrupted_accuracy": float(np.mean(pred["second_edge_corrupted_tokens"] == labels)),
        "second_edge_oracle_accuracy": float(np.mean(suite["second_edge_random_answers"] == labels)),
    }
    values = trajectory_value_statistics(tr, seed=seed, phase="matched_audit",
                                        positive_epsilon=study["development_gate"]["value_positive_epsilon"])
    gate = evaluate_seed_gate(values, causal, study["development_gate"])
    decisions = {
        "clean_correct": np.argmax(tr["probabilities"][:, -1], axis=1) == tr["labels"],
        "paired_cf_correct": pair_ok,
    }
    assert causal["counterfactual_pair_accuracy"] <= min(float(clean_ok.mean()), float(cf_ok.mean()))
    return tr, causal, values, gate, decisions


def run(out):
    started = time.time()
    manifest = json.loads((ROOT / "FROZEN_INPUT_SHA256.json").read_text())
    for name, digest in manifest.items():
        assert sha(ROOT / name) == digest, f"Input integrity failure: {name}"
    spec = json.loads((ROOT / "AUDIT_SPEC.json").read_text())
    assert spec["seeds"] == [503, 607, 709, 811, 907, 1009]
    assert spec["budget"]["training_updates"] == spec["budget"]["probe_fits"] == 0
    assert not out.exists(), "Use a new output directory; original results must not be overwritten."
    out.mkdir(parents=True)
    study = load_config(ROOT / "frozen/U7C/preregistered_protocol.yaml")
    b_study = load_config(ROOT / "frozen/U7B/preregistered_protocol.yaml")
    for key in ["model", "data", "development_gate"]:
        assert study[key] == b_study[key]
    assert spec["gate_thresholds"] == study["development_gate"]
    all_metrics, all_values, effects, fingerprints, native_checks = [], [], [], [], []
    unique_count = 0
    decisions_out = out / "per_example"
    decisions_out.mkdir()
    for seed in spec["seeds"]:
        models = {v: load_model(study, v, seed) for v in spec["models"]}
        offsets = {"original_U7B": (37, 43) if seed < 811 else (83, 89),
                   "original_U7C": (37, 43)}
        cache = {}
        for suite_name, (sample_offset, causal_offset) in offsets.items():
            key = (sample_offset, causal_offset)
            if key not in cache:
                clean = generate_clean_evaluation(study, seed, sample_offset)
                suite = generate_two_hop_counterfactual_suite(
                    2000, key_count=8, pair_count=5,
                    rng=np.random.default_rng(seed * 100 + causal_offset))
                assert clean["tokens"].shape == (4000, 12)
                assert suite["clean_tokens"].shape == (2000, 12)
                fp = {**{f"clean/{k}": array_sha(v) for k, v in clean.items()},
                      **{f"causal/{k}": array_sha(v) for k, v in suite.items()}}
                for arr in [*clean.values(), *suite.values()]:
                    arr.flags.writeable = False
                fingerprints.append({"seed": seed, "sample_offset": sample_offset,
                                     "causal_offset": causal_offset, "array_sha256": fp})
                cache[key] = {}
                for version, model in models.items():
                    tr, causal, values, gate, decisions = evaluate(model, study, seed, clean, suite)
                    if key == offsets["original_" + version]:
                        err = verify_native(tr, causal, values, gate, version, seed)
                        native_checks.append({"version": version, "seed": seed,
                                              "native_probability_max_absolute_error": err,
                                              "all_native_metrics_reproduced": True})
                    cache[key][version] = (causal, values, gate, decisions)
                    unique_count += 1
                    np.savez_compressed(decisions_out / f"{version}_seed_{seed}_offset_{sample_offset}.npz",
                                        **decisions)
                    del tr
                    print(f'{unique_count:02d}/18 {version} seed={seed} offsets={key} '
                          f'clean={gate["final_accuracy"]:.5f} paired={gate["counterfactual_pair_accuracy"]:.4f} '
                          f'gate={gate["passed"]}', flush=True)
                assert fp == {**{f"clean/{k}": array_sha(v) for k, v in clean.items()},
                              **{f"causal/{k}": array_sha(v) for k, v in suite.items()}}
            for version, (causal, values, gate, decisions) in cache[key].items():
                meta = {"version": version, "seed": seed, "suite": suite_name,
                        "sample_offset": sample_offset, "causal_offset": causal_offset,
                        "suite_is_alias": seed < 811 and suite_name == "original_U7C"}
                all_metrics.append({**meta, **causal, **gate})
                all_values.extend({**meta, **row} for row in values)
            b_decisions = cache[key]["U7B"][3]
            c_decisions = cache[key]["U7C"][3]
            row = {"seed": seed, "suite": suite_name, "suite_is_alias": seed < 811 and suite_name == "original_U7C"}
            for metric in ["clean_correct", "paired_cf_correct"]:
                b, c = b_decisions[metric], c_decisions[metric]
                wins, losses = int((c & ~b).sum()), int((b & ~c).sum())
                delta = (wins - losses) / len(b)
                np.testing.assert_allclose(delta, float(c.mean() - b.mean()), atol=1e-15)
                row.update({f"{metric}_n": len(b), f"{metric}_u7b": float(b.mean()),
                            f"{metric}_u7c": float(c.mean()), f"{metric}_delta": delta,
                            f"{metric}_c_only": wins, f"{metric}_b_only": losses})
            effects.append(row)
        # Progress tables are checkpoints, never substituted for a completed result.
        pd.DataFrame(all_metrics).to_csv(out / "matched_metrics.csv", index=False)
        pd.DataFrame(all_values).to_csv(out / "matched_value_statistics.csv", index=False)
        pd.DataFrame(effects).to_csv(out / "paired_effects.csv", index=False)
    assert unique_count == 18 and len(all_metrics) == 24 and len(native_checks) == 12
    for name, digest in manifest.items():
        assert sha(ROOT / name) == digest, f"Original input changed: {name}"
    frame = pd.DataFrame(all_metrics)
    decisions = []
    for (version, suite), group in frame.groupby(["version", "suite"]):
        decisions.append({"version": version, "suite": suite,
                          "passed_seeds": group.loc[group.passed, "seed"].tolist(),
                          "failed_seeds": group.loc[~group.passed, "seed"].tolist(),
                          "all_seed_rescue": bool(group.passed.all())})
    result = {"audit_id": spec["audit_id"], "status": "completed",
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.time() - started,
              "spec_sha256": sha(ROOT / "AUDIT_SPEC.json"),
              "runner_sha256": sha(Path(__file__)),
              "input_manifest_sha256": sha(ROOT / "FROZEN_INPUT_SHA256.json"),
              "unique_model_suite_evaluations": unique_count,
              "reported_model_suite_rows": len(all_metrics),
              "native_endpoint_checks": native_checks,
              "shared_arrays_unchanged_after_both_models": True,
              "frozen_inputs_unchanged": True,
              "training_updates": 0, "probe_fits": 0, "confirmation_seeds_opened": [],
              "original_failed_advancement_decisions_unchanged": True,
              "advance_to_t1_u8": False, "suite_decisions": decisions,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "platform": platform.platform()}}
    dump(out / "evaluation_fingerprints.json", fingerprints)
    dump(out / "audit_result.json", result)
    print("AUDIT COMPLETE: matched comparisons verified; original decisions preserved.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        run(args.out.resolve())
