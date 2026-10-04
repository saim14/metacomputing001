"""T2-E1: fixed exploratory ridge-probe study; no base-model training."""
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
from run_t1_u7 import load_config, generate_clean_evaluation
from src.t1_experiment import generate_task, generate_two_hop_counterfactual_suite
from src.tiny_transformer import ModelConfig, TinyTransformer, numpy_softmax

ARMS = ["current_linear", "current_dimension_control", "current_rff", "ordered_history",
        "shuffled_history", "sample_mismatched_history", "residualized_history"]
OBSERVERS = ["cls_current", "full_current"]
STEPS = [2, 3, 4, 5]
PARTS = {"train": slice(0, 3200), "validation": slice(3200, 4000), "test": slice(4000, 6000)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def family_keys(tokens):
    a = np.asarray(tokens, dtype=np.uint8)
    b = a.copy()
    b[a == 8] = 9
    b[a == 9] = 8
    return [min(x.tobytes(), y.tobytes()) for x, y in zip(a, b)]


def generate_data(study, seed):
    """Exclude exact families from all model streams and exposed evaluation sets."""
    forbidden = set()
    streams = {}
    for phase in study["training"]["curriculum"]:
        streams[phase["data_stream_id"]] = {**study["data"], **phase.get("data_overrides", {})}
    for stream_id, cfg in streams.items():
        for n, offset in [(cfg["train_samples"], 11), (cfg["validation_samples"], 23)]:
            tokens, _ = generate_task(n, data_config=cfg,
                rng=np.random.default_rng(seed * 10000 + stream_id * 100 + offset))
            forbidden.update(family_keys(tokens))
    for clean_offset, causal_offset in ([(37, 43)] if seed < 811 else [(37, 43), (83, 89)]):
        forbidden.update(family_keys(generate_clean_evaluation(study, seed, clean_offset)["tokens"]))
        suite = generate_two_hop_counterfactual_suite(2000, key_count=8, pair_count=5,
            rng=np.random.default_rng(seed * 100 + causal_offset))
        for name, array in suite.items():
            if name.endswith("tokens"):
                forbidden.update(family_keys(array))
    forbidden_count = len(forbidden)
    rng = np.random.default_rng(200000000 + seed)
    seen = set()
    chosen, labels = [], []
    rejected = 0
    for batch in range(5):
        suite = generate_two_hop_counterfactual_suite(4000, key_count=8, pair_count=5, rng=rng)
        for i, key in enumerate(family_keys(suite["clean_tokens"])):
            if key in forbidden or key in seen:
                rejected += 1
                continue
            seen.add(key)
            chosen.append(np.stack([suite["clean_tokens"][i], suite["counterfactual_tokens"][i]]))
            labels.append([suite["clean_labels"][i], suite["counterfactual_labels"][i]])
            if len(chosen) == 3000:
                break
        if len(chosen) == 3000:
            break
    assert len(chosen) == 3000, "Frozen candidate budget exhausted"
    order = np.random.default_rng(210000000 + seed).permutation(3000)
    tokens = np.asarray(chosen)[order].reshape(6000, 12)
    y = np.asarray(labels, dtype=np.int64)[order].reshape(6000)
    keys = family_keys(tokens)
    groups = [set(keys[p][::2]) for p in PARTS.values()]
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i+1:])
    assert all(keys[i] == keys[i+1] for i in range(0, 6000, 2))
    assert set(keys).isdisjoint(forbidden)
    assert len(set(keys)) == 3000
    for part in PARTS.values():
        assert y[part].mean() == 0.5
    metadata = {"seed": seed, "forbidden_unique_families": forbidden_count,
        "new_unique_families": len(seen), "candidate_batches_used": batch + 1,
        "rejected_candidates": rejected, "exact_family_overlap_with_excluded_data": 0,
        "cross_split_family_overlap": 0, "pair_members_share_split": True,
        "tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(y.tobytes()).hexdigest()}
    return tokens, y, metadata


def load_model(study, seed):
    model = TinyTransformer(ModelConfig(vocab_size=11, sequence_length=12, **study["model"]), seed=seed)
    with np.load(ROOT / "frozen/checkpoints" / f"seed_{seed}.npz", allow_pickle=False) as z:
        parameters = {p.name: p for p in model.parameters()}
        assert set(parameters) == set(z.files)
        for name, p in parameters.items():
            assert p.data.shape == z[name].shape
            p.data[...] = z[name]
            p.requires_grad = False
            p.grad = None
    return model


def extract(model, tokens):
    n = len(tokens)
    full = np.empty((n, 5, 384), np.float32)
    cls = np.empty((n, 7, 32), np.float32)
    probs = np.empty((n, 7, 2), np.float32)
    for start in range(0, n, 600):
        stop = min(n, start + 600)
        states, logits, _ = model(tokens[start:stop])
        for step in range(7):
            cls[start:stop, step] = states[step].data[:, 0]
            probs[start:stop, step] = numpy_softmax(logits[step].data)
            if 1 <= step <= 5:
                full[start:stop, step-1] = states[step].data.reshape(stop-start, 384)
    clipped = np.clip(probs, 1e-8, 1.0)
    entropy = -(clipped * np.log2(clipped)).sum(axis=-1, keepdims=True)
    margin = np.abs(probs[:, :, 1:2] - probs[:, :, 0:1])
    summaries = np.concatenate([probs, entropy, margin], axis=-1)
    features = np.concatenate([cls, summaries], axis=-1)
    assert features.shape == (n, 7, 36)
    assert np.isfinite(full).all() and np.isfinite(features).all()
    for step in STEPS:
        np.testing.assert_array_equal(full[:, step-1, :32], cls[:, step])
    return full, features, summaries, probs


def standardize(x):
    x = np.asarray(x, np.float64)
    mean = x[PARTS["train"]].mean(axis=0)
    sd = np.maximum(x[PARTS["train"]].std(axis=0), 1e-8)
    return (x - mean) / sd, mean, sd


def ridge_path(x, y, alphas):
    """Train-only preprocessing; validation chooses alpha; test is never consulted."""
    x, mean, sd = standardize(x)
    train, val = PARTS["train"], PARTS["validation"]
    y_mean = float(y[train].mean())
    gram = x[train].T @ x[train]
    vals, vectors = np.linalg.eigh(gram)
    vals = np.maximum(vals, 0.0)
    rhs = vectors.T @ (x[train].T @ (y[train] - y_mean))
    coefficients = vectors @ (rhs[:, None] / (vals[:, None] + np.asarray(alphas)[None, :]))
    val_pred = x[val] @ coefficients + y_mean
    val_mse = np.mean((val_pred - y[val, None])**2, axis=0)
    best = min(range(len(alphas)), key=lambda i: (val_mse[i], -alphas[i]))
    prediction = x @ coefficients[:, best] + y_mean
    return prediction, {"alpha": float(alphas[best]), "validation_mse": float(val_mse[best]),
        "train_mse": float(np.mean((prediction[train] - y[train])**2)),
        "fitted_coefficients_including_intercept": x.shape[1] + 1,
        "alpha_validation_mse": val_mse.tolist()}


def history_controls(features, seed, step):
    raw = features[:, 1:step].astype(np.float64)
    n, slots, d = raw.shape
    shuffled = raw.copy()
    rng = np.random.default_rng(220000000 + seed * 1000 + step * 100 + 70)
    if slots >= 2:
        for family in range(n // 2):
            perm = rng.permutation(slots)
            shuffled[2*family:2*family+2] = raw[2*family:2*family+2][:, perm]
    mismatched = raw.copy()
    for part in PARTS.values():
        group = raw[part].reshape(-1, 2, slots, d)
        shift = int(rng.integers(1, len(group)))
        donors = (np.arange(len(group)) + shift) % len(group)
        assert np.all(donors != np.arange(len(group)))
        mismatched[part] = group[donors].reshape(-1, slots, d)
    # A temporal shuffle preserves each example's past-step multiset.
    np.testing.assert_allclose(shuffled.sum(axis=1), raw.sum(axis=1), rtol=1e-12, atol=1e-12)
    return {"ordered_history": raw.reshape(n, -1), "shuffled_history": shuffled.reshape(n, -1),
            "sample_mismatched_history": mismatched.reshape(n, -1)}


def features_for_observer(current, histories, seed, step, observer):
    z, _, _ = standardize(current)
    n, d = current.shape
    h = histories["ordered_history"].shape[1]
    rng = np.random.default_rng(220000000 + seed * 1000 + step * 100 + observer)
    linear = z @ (rng.normal(size=(d, h)) / np.sqrt(d))
    w = rng.normal(size=(d, h)) / np.sqrt(d)
    bias = rng.uniform(0, 2*np.pi, size=h)
    rff = np.sqrt(2.0/h) * np.cos(z @ w + bias)
    past = histories["ordered_history"]
    train = PARTS["train"]
    past_mean = past[train].mean(axis=0)
    coef = np.linalg.solve(z[train].T @ z[train] + np.eye(d),
                           z[train].T @ (past[train] - past_mean))
    residual = past - (z @ coef + past_mean)
    appended = {**histories, "current_dimension_control": linear,
                "current_rff": rff, "residualized_history": residual}
    matrices = {"current_linear": np.asarray(current, np.float64),
                **{arm: np.concatenate([current, value], axis=1) for arm, value in appended.items()}}
    for arm in ARMS[1:]:
        assert matrices[arm].shape[1] == d + 36*(step-1)
        np.testing.assert_array_equal(matrices[arm][:, :d], current)
    return matrices


def r2(y, pred):
    sst = float(np.sum((y - y.mean())**2))
    return float(1 - np.sum((y-pred)**2) / sst) if sst > 1e-12 else None


def bootstrap_comparisons(out, spec):
    records, boot = [], {}
    rng = np.random.default_rng(spec["aggregation"]["bootstrap_rng"])
    for seed in spec["model_seeds"]:
        with np.load(out / "predictions" / f"seed_{seed}.npz", allow_pickle=False) as saved:
            y, pred, eligible = saved["target"], saved["prediction"], saved["eligible"]
        # One common family bootstrap per seed, shared across every comparison and step.
        counts = rng.multinomial(1000, np.full(1000, 0.001), size=1000).astype(np.float64)
        ys = y.reshape(1000, 2, 4).sum(axis=1)
        y2s = (y*y).reshape(1000, 2, 4).sum(axis=1)
        sst = counts @ y2s - (counts @ ys)**2 / 2000
        safe_sst = np.where(sst > 1e-12, sst, np.nan)
        for oi, observer in enumerate(OBSERVERS):
            candidate = pred[:, :, oi, ARMS.index("ordered_history")]
            for baseline in [a for a in ARMS if a != "ordered_history"]:
                indices = np.flatnonzero(eligible & (np.asarray(STEPS) >= (3 if baseline == "shuffled_history" else 2)))
                if not len(indices):
                    records.append({"seed": seed, "observer": observer, "baseline": baseline,
                                    "eligible_steps": [], "delta_r2": None, "mae_advantage": None})
                    continue
                assert (sst[:, indices] > 1e-12).all(), "Degenerate eligible held-out bootstrap target"
                current = pred[:, :, oi, ARMS.index(baseline)]
                sqdiff = ((y-current)**2 - (y-candidate)**2).reshape(1000, 2, 4).sum(axis=1)
                absdiff = (np.abs(y-current) - np.abs(y-candidate)).reshape(1000, 2, 4).sum(axis=1)
                point = np.mean(sqdiff.sum(axis=0)[indices] / np.sum((y-y.mean(axis=0))**2, axis=0)[indices])
                mae = float(absdiff[:, indices].mean()/2)
                bs = (counts @ sqdiff / safe_sst)[:, indices].mean(axis=1)
                bmae = (counts @ absdiff / 2000)[:, indices].mean(axis=1)
                records.append({"seed": seed, "observer": observer, "baseline": baseline,
                    "eligible_steps": [STEPS[i] for i in indices], "delta_r2": float(point),
                    "mae_advantage": mae, "conditional_ci95": np.quantile(bs, [.025, .975]).tolist()})
                boot[(seed, observer, baseline)] = (bs, bmae)
    summary = []
    for observer in OBSERVERS:
        for baseline in [a for a in ARMS if a != "ordered_history"]:
            relevant = [r for r in records if r["observer"] == observer and r["baseline"] == baseline and r["delta_r2"] is not None]
            if not relevant:
                continue
            boots = [boot[(r["seed"], observer, baseline)] for r in relevant]
            bs = np.mean([b[0] for b in boots], axis=0)
            row = {"observer": observer, "baseline": baseline,
                "contributing_seeds": [r["seed"] for r in relevant],
                "mean_delta_r2": float(np.mean([r["delta_r2"] for r in relevant])),
                "mean_mae_advantage": float(np.mean([r["mae_advantage"] for r in relevant])),
                "conditional_ci95": np.quantile(bs, [.025, .975]).tolist(),
                "positive_seed_count": sum(r["delta_r2"] > 0 for r in relevant)}
            row["passes_frozen_screen"] = bool(len(relevant) == 6 and row["mean_delta_r2"] >= .01
                and row["conditional_ci95"][0] > 0 and row["positive_seed_count"] == 6
                and row["mean_mae_advantage"] >= 0)
            summary.append(row)
    dump(out / "seed_comparisons.json", records)
    dump(out / "panel_comparisons.json", summary)
    primary = [r for r in summary if r["observer"] == "full_current" and r["baseline"] in ["current_rff", "current_linear"]]
    return records, summary, len(primary) == 2 and all(r["passes_frozen_screen"] for r in primary)


def run(out):
    started = time.time()
    manifest = json.loads((ROOT / "FROZEN_INPUT_SHA256.json").read_text())
    for name, digest in manifest.items():
        assert sha(ROOT/name) == digest, name
    spec = json.loads((ROOT / "STUDY_PROTOCOL.json").read_text())
    assert spec["model_seeds"] == [503, 607, 709, 811, 907, 1009]
    assert spec["arms"] == ARMS and spec["steps"] == STEPS
    study = load_config(ROOT / "frozen/model_protocol.yaml")
    assert not out.exists(), "A new output directory is required"
    out.mkdir(parents=True)
    for name in ["data", "predictions"]:
        (out/name).mkdir()
    metrics, diagnostics, data_checks, tuning = [], [], [], []
    fit_count = 0
    for seed in spec["model_seeds"]:
        seed_started = time.time()
        print(f"PREPARE seed={seed}: fresh grouped examples; frozen model", flush=True)
        tokens, labels, data_check = generate_data(study, seed)
        data_checks.append(data_check)
        np.savez_compressed(out / "data" / f"seed_{seed}.npz", tokens=tokens, labels=labels)
        full, cls_features, summaries, probs = extract(load_model(study, seed), tokens)
        true_probs = probs.astype(np.float64)[np.arange(6000)[:, None], np.arange(7)[None, :], labels[:, None]]
        nll = -np.log(np.clip(true_probs, 1e-8, 1.0))
        target = nll[:, STEPS] - nll[:, -1, None]
        eligible = np.asarray([(target[PARTS["train"], i].mean() >= .02 and
                               target[PARTS["train"], i].std(ddof=1) >= .02) for i in range(4)])
        test_predictions = np.empty((2000, 4, 2, 7), np.float64)
        print(f"EXTRACTED seed={seed}: eligible steps={np.asarray(STEPS)[eligible].tolist()}", flush=True)
        for si, step in enumerate(STEPS):
            y = target[:, si]
            diagnostics.append({"seed": seed, "step": step, "eligible": bool(eligible[si]),
                "train_mean_value": float(y[PARTS["train"]].mean()), "train_sd_value": float(y[PARTS["train"]].std(ddof=1)),
                "test_mean_value": float(y[PARTS["test"]].mean()), "test_sd_value": float(y[PARTS["test"]].std(ddof=1)),
                "test_final_accuracy": float(np.mean(probs[PARTS["test"], -1].argmax(axis=1) == labels[PARTS["test"]])),
                "test_flip_prevalence": float(np.mean(probs[PARTS["test"], step].argmax(axis=1) != probs[PARTS["test"], -1].argmax(axis=1)))})
            histories = history_controls(cls_features, seed, step)
            for oi, observer in enumerate(OBSERVERS):
                current = cls_features[:, step] if observer == "cls_current" else np.concatenate([full[:, step-1], summaries[:, step]], axis=1)
                matrices = features_for_observer(current, histories, seed, step, oi)
                fitted = {}
                for ai, arm in enumerate(ARMS):
                    alias = step == 2 and arm == "shuffled_history"
                    fit_started = time.time()
                    if alias:
                        prediction, details = fitted["ordered_history"]
                    else:
                        prediction, details = ridge_path(matrices[arm], y, spec["probe"]["alpha_grid"])
                        fit_count += 1
                    fitted[arm] = (prediction, details)
                    test_predictions[:, si, oi, ai] = prediction[PARTS["test"]]
                    yy, pp = y[PARTS["test"]], prediction[PARTS["test"]]
                    row = {"seed": seed, "step": step, "observer": observer, "arm": arm,
                        "eligible": bool(eligible[si]), "alias_of_ordered": alias,
                        **{k: v for k, v in details.items() if k != "alpha_validation_mse"},
                        "test_r2": r2(yy, pp), "test_mae": float(np.abs(yy-pp).mean()),
                        "test_mse": float(np.mean((yy-pp)**2)), "fit_seconds": time.time()-fit_started}
                    metrics.append(row)
                    tuning.append({"seed": seed, "step": step, "observer": observer, "arm": arm,
                                   "validation_mse_by_alpha": details["alpha_validation_mse"], "selected_alpha": details["alpha"]})
                del matrices, fitted
        np.savez_compressed(out / "predictions" / f"seed_{seed}.npz",
            target=target[PARTS["test"]], prediction=test_predictions, eligible=eligible)
        pd.DataFrame(metrics).to_csv(out / "probe_metrics.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(out / "target_diagnostics.csv", index=False)
        dump(out / "data_checks.json", data_checks)
        dump(out / "validation_selection.json", tuning)
        print(f"FINISHED seed={seed} elapsed={time.time()-seed_started:.1f}s fits={fit_count}", flush=True)
        del full, cls_features, summaries, probs, target
    assert len(metrics) == 336 and fit_count == 324
    records, panel, passed = bootstrap_comparisons(out, spec)
    for name, digest in manifest.items():
        assert sha(ROOT/name) == digest, name
    result = {"study_id": "T2-E1", "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time()-started, "training_updates": 0, "reserved_model_seeds_opened": [],
        "actual_probe_fits": fit_count, "alpha_candidates_evaluated": fit_count*5,
        "reported_cells": len(metrics), "new_probe_examples": 36000,
        "protocol_sha256": sha(ROOT/"STUDY_PROTOCOL.json"), "runner_sha256": sha(__file__),
        "frozen_inputs_unchanged": True, "all_six_exposed_models_reported": True,
        "primary_exploratory_screen_passed": bool(passed), "confirmation_advancement": False,
        "uncertainty_scope": spec["aggregation"]["uncertainty"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    dump(out / "study_result.json", result)
    print("STUDY COMPLETE. Exploratory screen passed:", passed, flush=True)
    for row in panel:
        if row["baseline"] in ["current_rff", "current_linear"]:
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        run(args.out.resolve())
