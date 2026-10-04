"""Frozen P1-A1 v1.1 development observers. No network training path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "inputs/prior"))
import experiment as base

ARMS = ["current", "expanded_current", "ordered_history", "residual_history",
        "mismatched_history", "shuffled_history"]
SYNTHETIC_ARMS = ["current", "expanded_current", "ordered_history", "mismatched_history"]
SPLITS = [314159, 314160, 314161]
MODELS = [2017, 2027]
STEPS = [3, 4]
PENALTIES = [1.0, 0.01, 0.0001]
KINDS = [("linear", None), ("rbf", 0.5), ("rbf", 2.0)]
DELTA = 0.01
BOOTSTRAPS = 5000


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_for(*parts):
    value = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(value[:8], "little"))


def family_weights(fids):
    _, inverse, counts = np.unique(fids, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse])


def family_means(values, fids):
    ids, inverse, counts = np.unique(fids, return_inverse=True, return_counts=True)
    return ids, np.bincount(inverse, weights=values) / counts


def fit_scale(x, w):
    mean = w @ x
    sd = np.sqrt(w @ ((x - mean) ** 2))
    sd = np.where(sd < 1e-8, 1.0, sd)
    return {"mean": mean, "sd": sd}


def scale(x, state):
    return (x - state["mean"]) / state["sd"]


def ridge_grid(x, y, w, penalties):
    """Weighted MSE + lambda ||beta||^2, unpenalized intercept."""
    xm, ym = w @ x, w @ y
    xc, yc = x - xm, y - ym
    gram = xc.T @ (w[:, None] * xc)
    rhs = xc.T @ (w[:, None] * yc) if y.ndim == 2 else xc.T @ (w * yc)
    eigenvalues, q = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = q.T @ rhs
    rows = []
    for penalty in penalties:
        divisor = eigenvalues + penalty
        beta = q @ (projected / (divisor[:, None] if y.ndim == 2 else divisor))
        intercept = ym - xm @ beta
        assert np.all(np.isfinite(beta)) and np.all(np.isfinite(intercept))
        rows.append({"coef": beta, "intercept": intercept, "penalty": penalty})
    return rows


def one_observation(trajectory, step):
    state = trajectory["states"][:, step]
    probability = trajectory["probabilities"][:, step].astype(np.float64)
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(1, keepdims=True)
    ordered = np.sort(probability, axis=1)
    margin = (ordered[:, -1] - ordered[:, -2])[:, None]
    return np.concatenate([state.reshape(len(state), -1), probability, entropy, margin], axis=1)


def features(trajectory, step):
    assert step in STEPS
    return one_observation(trajectory, step), np.concatenate(
        [one_observation(trajectory, j) for j in range(step)], axis=1)


def target(trajectory, step):
    return (trajectory["logits"][:, step].argmax(1) !=
            trajectory["logits"][:, -1].argmax(1)).astype(np.float64)


def observer_split(train_fids, split_seed):
    ids = np.unique(train_fids)
    assert len(ids) == 581
    ordered = np.random.default_rng(split_seed).permutation(ids)
    return {"fit": np.flatnonzero(np.isin(train_fids, ordered[:465])),
            "tune": np.flatnonzero(np.isin(train_fids, ordered[465:]))}


def support(y, fids, partition):
    counts = [int(np.sum(y == cls)) for cls in (0, 1)]
    families = [int(len(np.unique(fids[y == cls]))) for cls in (0, 1)]
    min_examples, min_families = (50, 25) if partition == "fit" else (20, 10)
    return {"counts": counts, "families": families,
            "minority_example_fraction": min(counts) / len(y),
            "eligible": min(counts) >= min_examples and min(families) >= min_families}


def donor_map(fids, rng):
    groups = rng.permutation(np.unique(fids))
    blocks = [rng.permutation(np.flatnonzero(fids == g)) for g in groups]
    order = np.concatenate(blocks)
    shift = max(map(len, blocks))
    assert len(order) > 2 * shift
    donor = np.empty(len(fids), dtype=np.int64)
    donor[order] = np.roll(order, shift)
    assert np.array_equal(np.sort(donor), np.arange(len(fids)))
    assert np.all(fids[donor] != fids)
    return donor


def history_control(h, fids, step, arm, key):
    if arm == "mismatched_history":
        donor = donor_map(fids, rng_for(*key, "donor"))
        return h[donor], {"donor": donor}
    if arm == "shuffled_history":
        rng = rng_for(*key, "shuffle")
        orders = np.stack([rng.permutation(step) for _ in range(len(h))])
        blocks = h.reshape(len(h), step, -1)
        result = np.take_along_axis(blocks, orders[:, :, None], axis=1).reshape(h.shape)
        return result, {"orders": orders}
    return h, {}


def fit_preprocessor(c, h, fids, arm, key):
    w = family_weights(fids)
    state = {"arm": arm, "c_scale": fit_scale(c, w), "history_width": h.shape[1]}
    zc = scale(c, state["c_scale"])
    if arm == "current":
        return state
    if arm == "expanded_current":
        rng = rng_for(*key, "cosine")
        state["W"] = rng.normal(size=(c.shape[1], h.shape[1])) / np.sqrt(c.shape[1])
        state["b"] = rng.uniform(0, 2 * np.pi, size=h.shape[1])
        addition = np.sqrt(2.0) * np.cos(zc @ state["W"] + state["b"])
    else:
        state["h_scale"] = fit_scale(h, w)
        addition = scale(h, state["h_scale"])
        if arm == "residual_history":
            residual_fit = ridge_grid(zc, addition, w, [0.01])[0]
            state["res_coef"] = residual_fit["coef"]
            state["res_intercept"] = residual_fit["intercept"]
            addition = addition - zc @ state["res_coef"] - state["res_intercept"]
    state["addition_scale"] = fit_scale(addition, w)
    return state


def transform(c, h, state):
    zc = scale(c, state["c_scale"])
    arm = state["arm"]
    current = zc / np.sqrt(zc.shape[1])
    if arm == "current":
        return current
    if arm == "expanded_current":
        addition = np.sqrt(2.0) * np.cos(zc @ state["W"] + state["b"])
    else:
        addition = scale(h, state["h_scale"])
        if arm == "residual_history":
            addition = addition - zc @ state["res_coef"] - state["res_intercept"]
    addition = scale(addition, state["addition_scale"]) / np.sqrt(addition.shape[1])
    return np.concatenate([current, addition], axis=1) / np.sqrt(2.0)


def landmark_indices(fids, key):
    rng = rng_for(*key, "landmarks")
    families = rng.permutation(np.unique(fids))[:64]
    assert len(families) == 64
    return np.array([rng.choice(np.flatnonzero(fids == f)) for f in families], dtype=np.int64)


def gaussian(z, landmarks, bandwidth):
    distance = np.maximum((z * z).sum(1)[:, None] + (landmarks * landmarks).sum(1)[None, :]
                          - 2 * z @ landmarks.T, 0.0)
    return np.exp(-distance / (2 * bandwidth ** 2))


def predict(pipe, c, h):
    z = transform(c, h, pipe["preprocessor"])
    if pipe["kind"] == "rbf":
        z = scale(gaussian(z, pipe["landmarks"], pipe["bandwidth"]), pipe["phi_scale"]) / 8.0
    return np.clip(z @ pipe["coef"] + pipe["intercept"], 0, 1)


def metrics(y, p, fids):
    w = family_weights(fids)
    brier = float(w @ ((p - y) ** 2))
    positives, negatives = float(w @ y), float(w @ (1 - y))
    thresholded = p >= 0.5
    balanced = 0.5 * (float(w @ ((y == 1) & thresholded)) / positives +
                      float(w @ ((y == 0) & ~thresholded)) / negatives) if min(positives, negatives) > 0 else None
    _, inverse = np.unique(p, return_inverse=True)
    pos = np.bincount(inverse, weights=w * y)
    neg = np.bincount(inverse, weights=w * (1 - y))
    auc = float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (positives * negatives)) if min(positives, negatives) > 0 else None
    return {"brier": brier, "balanced_accuracy": balanced, "roc_auc": auc,
            "family_weighted_prevalence": positives}


def save_tree(path, obj):
    arrays = {}
    def visit(value):
        if isinstance(value, np.ndarray):
            key = f"a{len(arrays)}"
            arrays[key] = value
            return {"__array__": key}
        if isinstance(value, dict):
            return {str(k): visit(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value
    structure = visit(obj)
    arrays["__structure__"] = np.array(json.dumps(structure, allow_nan=False))
    np.savez_compressed(path, **arrays)


def load_tree(path):
    with np.load(path, allow_pickle=False) as arrays:
        def visit(value):
            if isinstance(value, dict):
                if set(value) == {"__array__"}:
                    return arrays[value["__array__"]].copy()
                return {k: visit(v) for k, v in value.items()}
            if isinstance(value, list):
                return [visit(v) for v in value]
            return value
        return visit(json.loads(str(arrays["__structure__"])))


def fit_arm(data, arm, step, key):
    fit, tune = data["fit"], data["tune"]
    w = family_weights(fit["fids"])
    wt = family_weights(tune["fids"])
    pre = fit_preprocessor(fit["c"], fit["h"], fit["fids"], arm, key)
    z = transform(fit["c"], fit["h"], pre)
    zt = transform(tune["c"], tune["h"], pre)
    indices = landmark_indices(fit["fids"], key)
    candidates, pipes, tuning_predictions = [], [], []
    for kind, bandwidth in KINDS:
        extras = {"kind": kind, "bandwidth": bandwidth}
        if kind == "linear":
            x, xt = z, zt
        else:
            landmarks = z[indices].copy()
            phi = gaussian(z, landmarks, bandwidth)
            phi_scale = fit_scale(phi, w)
            x = scale(phi, phi_scale) / 8.0
            xt = scale(gaussian(zt, landmarks, bandwidth), phi_scale) / 8.0
            extras.update({"landmarks": landmarks, "landmark_indices": indices, "phi_scale": phi_scale})
        for fitted in ridge_grid(x, fit["y"], w, PENALTIES):
            p_tune = np.clip(xt @ fitted["coef"] + fitted["intercept"], 0, 1)
            tuning_predictions.append(p_tune)
            pipe = {"preprocessor": pre, **extras, **fitted}
            pipes.append(pipe)
            candidates.append({"index": len(candidates), "kind": kind, "bandwidth": bandwidth,
                "penalty": fitted["penalty"], "coefficient_count": len(fitted["coef"]) + 1,
                "input_width": z.shape[1], "tune_brier": float(wt @ ((p_tune-tune["y"]) ** 2)),
                "fit_brier": float(w @ ((np.clip(x @ fitted["coef"]+fitted["intercept"],0,1)-fit["y"])**2))})
    selected = min(range(9), key=lambda i: (candidates[i]["tune_brier"], i))
    return pipes[selected], {"candidates": candidates, "selected_index": selected,
                            "selected": candidates[selected], "tuning_predictions": np.stack(tuning_predictions)}


def timed(function):
    for _ in range(3):
        function()
    seconds = []
    for _ in range(15):
        start = time.perf_counter()
        function()
        seconds.append(time.perf_counter() - start)
    return {"median_seconds": float(np.median(seconds)), "repetitions_seconds": seconds,
            "warmup_repetitions": 3}


def run_cell(data, step, key, directory, arms=ARMS, do_timing=False):
    directory.mkdir(exist_ok=False)
    result = {"key": list(key), "step": step, "started_at_utc": now(), "arms": {}}
    predictions = {"y": data["eval"]["y"], "fids": data["eval"]["fids"]}
    tuning_predictions = {"y": data["tune"]["y"], "fids": data["tune"]["fids"]}
    controls = {}
    for arm in arms:
        arm_data, maps = {}, {}
        for partition, values in data.items():
            changed, record = history_control(values["h"], values["fids"], step, arm, (*key, partition))
            arm_data[partition] = {**values, "h": changed}
            maps[partition] = record
        controls[arm] = maps
        pipe, record = fit_arm(arm_data, arm, step, key)
        tuning_predictions[arm] = record.pop("tuning_predictions")
        evaluation = arm_data["eval"]
        p = predict(pipe, evaluation["c"], evaluation["h"])
        assert np.all(np.isfinite(p))
        record["evaluation"] = metrics(evaluation["y"], p, evaluation["fids"])
        save_tree(directory / f"pipeline_{arm}.npz", pipe)
        predictions[arm] = p
        result["arms"][arm] = record
    reference = min(["current", "expanded_current"], key=lambda a: (
        result["arms"][a]["selected"]["tune_brier"], a != "current"))
    result["current_reference"] = reference
    baseline = float(family_weights(data["fit"]["fids"]) @ data["fit"]["y"])
    predictions["baseline"] = np.full(len(data["eval"]["y"]), baseline)
    result["baseline_prevalence"] = baseline
    result["baseline_evaluation"] = metrics(data["eval"]["y"], predictions["baseline"], data["eval"]["fids"])
    if do_timing:
        result["timing"] = {}
        for arm in (reference, "ordered_history"):
            pipe = load_tree(directory / f"pipeline_{arm}.npz")
            result["timing"][arm] = timed(lambda: predict(pipe, data["eval"]["c"], data["eval"]["h"]))
    result["target_fits"] = len(arms) * 9
    result["completed_at_utc"] = now()
    save_tree(directory / "control_maps.npz", controls)
    np.savez_compressed(directory / "predictions.npz", **predictions)
    np.savez_compressed(directory / "tuning_predictions.npz", **tuning_predictions)
    write_json(directory / "result.json", result)
    return result


def bootstrap_interval(values, draws):
    sampled = values[draws].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
            "lower95": float(np.quantile(sampled, 0.05)), "upper95": float(np.quantile(sampled, 0.95))}


def summarize_cell(directory, draws):
    result = json.loads((directory / "result.json").read_text())
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    fids, y = arrays["fids"], arrays["y"]
    losses = {arm: family_means((arrays[arm]-y)**2, fids)[1]
              for arm in [*result["arms"], "baseline"]}
    c = result["current_reference"]
    delta = losses[c] - losses["ordered_history"]
    utility = losses["baseline"] - losses[c]
    delta_stats = bootstrap_interval(delta, draws)
    utility_stats = bootstrap_interval(utility, draws)
    skill = 1 - losses[c].mean() / losses["baseline"].mean()
    useful = skill >= 0.1 and utility_stats["lower95"] > 0
    return {"key": result["key"], "current_reference": c,
            "history_advantage": delta_stats, "current_skill": float(skill),
            "baseline_improvement": utility_stats, "useful": bool(useful),
            "correspondence_advantage": bootstrap_interval(losses["mismatched_history"]-losses["ordered_history"], draws),
            "losses": {arm: float(value.mean()) for arm, value in losses.items()}}, {
                "delta": delta, "correspondence": losses["mismatched_history"]-losses["ordered_history"]}


def analyze(output):
    expected_fids = np.unique(np.load(ROOT / "inputs/development_data.npz")["calibration_family_ids"])
    draws = np.random.default_rng(73001).integers(0, len(expected_fids), size=(BOOTSTRAPS, len(expected_fids)))
    natural, by_split, arrays = [], {}, {}
    for split_seed in SPLITS:
        deltas, correspondence = [], []
        for model_seed in MODELS:
            for step in STEPS:
                directory = output / f"split_{split_seed}_model_{model_seed}_step_{step}"
                row, vectors = summarize_cell(directory, draws)
                natural.append(row)
                arrays[(split_seed, model_seed, step)] = vectors
                deltas.append(vectors["delta"])
                correspondence.append(vectors["correspondence"])
        by_split[str(split_seed)] = {"history_advantage": bootstrap_interval(np.mean(deltas, axis=0), draws),
                                    "correspondence_advantage": bootstrap_interval(np.mean(correspondence, axis=0), draws)}
    synthetic = []
    for seed in [9101, 9102, 9103, 9104]:
        for level in [0.0, 0.1, 0.2]:
            row, _ = summarize_cell(output / f"synthetic_{seed}_b{level:.1f}", draws)
            row.update({"synthetic_seed": seed, "history_amplitude": level, "ideal_latent_brier_gain": level ** 2})
            synthetic.append(row)
    strong = sum(r["history_advantage"]["lower95"] > 0 and r["history_advantage"]["mean"] >= DELTA
                 for r in synthetic if r["history_amplitude"] == 0.2)
    near = sum(r["history_advantage"]["lower95"] > 0 for r in synthetic if r["history_amplitude"] == 0.1)
    null_false = sum(r["history_advantage"]["lower95"] > DELTA for r in synthetic if r["history_amplitude"] == 0)
    sensitivity = {"strong_signal_successes": int(strong), "near_margin_successes": int(near),
                   "null_material_false_positives": int(null_false), "replicates_per_level": 4,
                   "passed": strong == 4 and near >= 3 and null_false == 0}
    primary = by_split[str(SPLITS[0])]
    primary_rows = [r for r in natural if r["key"][0] == SPLITS[0]]
    by_model = {str(seed): bootstrap_interval(np.mean([
        arrays[(SPLITS[0], seed, step)]["delta"] for step in STEPS], axis=0), draws) for seed in MODELS}
    history_checks = {
        "primary_pooled_lower_exceeds_margin": primary["history_advantage"]["lower95"] > DELTA,
        "positive_point_effect_in_both_models": all(r["mean"] > 0 for r in by_model.values()),
        "correspondence_lower_positive": primary["correspondence_advantage"]["lower95"] > 0,
        "other_splits_point_effect_exceeds_margin": all(by_split[str(s)]["history_advantage"]["mean"] > DELTA for s in SPLITS[1:])}
    sufficiency_checks = {
        "all_primary_cell_upper_bounds_below_margin": all(r["history_advantage"]["upper95"] < DELTA for r in primary_rows),
        "other_splits_pooled_upper_below_margin": all(by_split[str(s)]["history_advantage"]["upper95"] < DELTA for s in SPLITS[1:]),
        "all_primary_current_predictors_useful": all(r["useful"] for r in primary_rows),
        "synthetic_sensitivity_passed": sensitivity["passed"]}
    status = ("development_history_advantage" if all(history_checks.values()) else
              "bounded_practical_current_state_sufficiency" if all(sufficiency_checks.values()) else
              "inconclusive_at_fixed_budgets")
    cheap = []
    for seed in MODELS:
        for step in STEPS:
            r = json.loads((output / f"split_{SPLITS[0]}_model_{seed}_step_{step}/result.json").read_text())
            cheap.append(r["timing"][r["current_reference"]]["median_seconds"] < r["timing"]["continuation"]["median_seconds"])
    return {"status": status, "margin_absolute_brier": DELTA, "primary": primary,
            "by_model_primary": by_model, "by_observer_split": by_split, "natural_cells": natural,
            "synthetic_cells": synthetic, "sensitivity": sensitivity,
            "history_checks": history_checks, "sufficiency_checks": sufficiency_checks,
            "current_faster_than_continuation_all_primary_cells": bool(all(cheap)),
            "confirmation_opened": False, "awareness_tested": False,
            "uncertainty_scope": "Paired canonical-family bootstrap, conditional on two fixed models and exposed development split."}


def make_natural_data(trajectories, development, split_seed, model_seed, step):
    indices = observer_split(development["train_family_ids"], split_seed)
    c, h = features(trajectories[(model_seed, "train")], step)
    y = target(trajectories[(model_seed, "train")], step)
    data = {part: {"c": c[idx], "h": h[idx], "y": y[idx], "fids": development["train_family_ids"][idx]}
            for part, idx in indices.items()}
    ce, he = features(trajectories[(model_seed, "calibration")], step)
    data["eval"] = {"c": ce, "h": he, "y": target(trajectories[(model_seed, "calibration")], step),
                    "fids": development["calibration_family_ids"]}
    return data


def synthetic_data(development, seed, level):
    train_fids, eval_fids = development["train_family_ids"], development["calibration_family_ids"]
    fids = np.concatenate([train_fids, eval_fids])
    ids, inverse = np.unique(fids, return_inverse=True)
    rng = np.random.default_rng(seed)
    u, v = (rng.choice([-1.0, 1.0], size=len(ids))[inverse] for _ in range(2))
    c = rng.normal(size=(len(fids), 115))
    h = rng.normal(size=(len(fids), 460))
    c[:, :8] = u[:, None] + 0.1 * rng.normal(size=(len(fids), 8))
    h[:, :32] = v[:, None] + 0.1 * rng.normal(size=(len(fids), 32))
    uniform = rng.random(len(fids))
    probability = 0.5 + 0.2 * u + level * v
    y = (uniform < probability).astype(np.float64)
    indices = observer_split(train_fids, SPLITS[0])
    indices["eval"] = np.arange(len(train_fids), len(fids))
    return {part: {"c": c[idx], "h": h[idx], "y": y[idx], "fids": fids[idx]}
            for part, idx in indices.items()}


def scientific_files():
    paths = [ROOT / n for n in ["run_a1.py", "methods_a1.py", "P1_A1_Protocol.md", "INPUT_PROVENANCE.json", "AMENDMENT.md",
                              "stopped_v1/CONTENTS_SHA256.json"]]
    paths += [p for p in (ROOT / "inputs").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def run(mode, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    freeze = {"frozen_at_utc": now(), "files_sha256": scientific_files(), "mode": mode,
              "scope": "Local prospective freeze; no independent timestamp registration."}
    write_json(output / "FREEZE.json", freeze)
    start = time.perf_counter()
    try:
        from methods_a1 import method_checks
        checks = method_checks(output)
        write_json(output / "method_checks.json", checks)
        assert checks["passed"]
        if mode == "methods":
            print(json.dumps(checks, indent=2), flush=True)
            return
        with np.load(ROOT / "inputs/development_data.npz", allow_pickle=False) as saved:
            development = {name: saved[name] for name in saved.files}
        expected, partitions = base.make_partitions()
        for part, values in expected.items():
            for k, v in values.items():
                np.testing.assert_array_equal(v, development[f"{part}_{k}"])
        assert partitions == json.loads((ROOT / "inputs/partitions.json").read_text())
        assert partitions["reserve_examples_constructed"] is False
        split_records = {}
        for split_seed in SPLITS:
            split_records[str(split_seed)] = {k: development["train_family_ids"][v].tolist()
                for k, v in observer_split(development["train_family_ids"], split_seed).items()}
        write_json(output / "observer_splits.json", split_records)
        trajectories, models = {}, {}
        for seed in MODELS:
            model = base.ArithmeticTransformer(seed=seed)
            model.load(ROOT / f"inputs/model_{seed}.npz")
            models[seed] = model
            for part in ("train", "calibration"):
                tr = base.extract(model, development[f"{part}_tokens"])
                trajectories[(seed, part)] = tr
                np.savez_compressed(output / f"trajectory_{seed}_{part}.npz", **tr)
        eligibility = []
        for split_seed in SPLITS:
            for seed in MODELS:
                for step in STEPS:
                    data = make_natural_data(trajectories, development, split_seed, seed, step)
                    eligibility.append({"split": split_seed, "model": seed, "step": step,
                        **{part: support(data[part]["y"], data[part]["fids"], part) for part in ("fit", "tune")}})
        write_json(output / "eligibility.json", eligibility)
        assert all(r[part]["eligible"] for r in eligibility for part in ("fit", "tune")), "Planned observer panel is ineligible."
        assert scientific_files() == freeze["files_sha256"]
        write_json(output / "FITTING_STARTED.json", {"started_at_utc": now(), "natural_target_fit_budget": 648,
                                                     "synthetic_target_fit_budget": 432})
        completed = []
        for split_seed in SPLITS:
            for seed in MODELS:
                for step in STEPS:
                    data = make_natural_data(trajectories, development, split_seed, seed, step)
                    directory = output / f"split_{split_seed}_model_{seed}_step_{step}"
                    row = run_cell(data, step, (split_seed, seed, step), directory, do_timing=split_seed == SPLITS[0])
                    if split_seed == SPLITS[0]:
                        states = trajectories[(seed, "calibration")]["states"][:, step]
                        model = models[seed]
                        def continuation():
                            with base.inference(model):
                                return np.concatenate([model.resume(states[i:i+64], step) for i in range(0, len(states), 64)])
                        resumed = continuation()
                        expected_logits = trajectories[(seed, "calibration")]["logits"][:, -1]
                        np.testing.assert_allclose(resumed, expected_logits, atol=2e-5, rtol=2e-5)
                        row["timing"]["continuation"] = timed(continuation)
                        row["resume_max_error"] = float(np.max(np.abs(resumed - expected_logits)))
                        write_json(directory / "result.json", row)
                    completed.append({"directory": directory.name, "target_fits": row["target_fits"]})
                    write_json(output / "completed_cells.json", completed)
                    print(f"{directory.name}: reference={row['current_reference']}, "
                          f"Brier current={row['arms'][row['current_reference']]['evaluation']['brier']:.5f}, "
                          f"history={row['arms']['ordered_history']['evaluation']['brier']:.5f}", flush=True)
        for seed in [9101, 9102, 9103, 9104]:
            for level in [0.0, 0.1, 0.2]:
                data = synthetic_data(development, seed, level)
                directory = output / f"synthetic_{seed}_b{level:.1f}"
                row = run_cell(data, 4, ("synthetic", seed, level), directory, arms=SYNTHETIC_ARMS)
                completed.append({"directory": directory.name, "target_fits": row["target_fits"]})
                write_json(output / "completed_cells.json", completed)
                print(f"{directory.name}: completed 36 candidate fits", flush=True)
        assert sum(r["target_fits"] for r in completed) == 1080
        assert scientific_files() == freeze["files_sha256"]
        decision = analyze(output)
        result = {"study_id": "P1-A1", "version": "1.1.0", "completed_at_utc": now(),
                  "elapsed_seconds": time.perf_counter() - start, "natural_target_fits": 648,
                  "synthetic_target_fits": 432, "arithmetic_network_training_updates": 0,
                  "reserved_examples_evaluated": 0, "frozen_inputs_unchanged": True, "decision": decision,
                  "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()}}
        write_json(output / "result.json", result)
        print(json.dumps({"status": decision["status"], "primary": decision["primary"],
                          "sensitivity": decision["sensitivity"]}, indent=2), flush=True)
    except Exception as error:
        write_json(output / "INCOMPLETE.json", {"at_utc": now(), "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["methods", "panel"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.mode, args.output)
