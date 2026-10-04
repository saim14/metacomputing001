"""T2-C1: one fixed synthetic calibration, without Transformer training."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
ARMS = ["current_linear", "current_rff", "history"]
DELTAS = np.array([0., .01, .05, .10])
ALPHAS = np.array([.01, .1, 1., 10., 100.])
TRAIN, VAL, TEST = slice(0, 3200), slice(3200, 4000), slice(4000, 6000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def correlated_pairs(rng, dimensions):
    family = rng.normal(size=(3000, 1, dimensions))
    member = rng.normal(size=(3000, 2, dimensions))
    return ((family + member) / np.sqrt(2.)).reshape(6000, dimensions)


def standardize(x):
    mean = x[TRAIN].mean(axis=0)
    sd = np.maximum(x[TRAIN].std(axis=0), 1e-8)
    return (x - mean) / sd


def ridge_targets(x, targets):
    """Reuse the X decomposition; each target has its own training and alpha choice."""
    z = standardize(x)
    mean_y = targets[TRAIN].mean(axis=0)
    vals, vecs = np.linalg.eigh(z[TRAIN].T @ z[TRAIN])
    vals = np.maximum(vals, 0.)
    rhs = vecs.T @ (z[TRAIN].T @ (targets[TRAIN] - mean_y))
    paths = rhs[:, :, None] / (vals[:, None, None] + ALPHAS[None, None, :])
    coef = (vecs @ paths.reshape(len(vals), -1)).reshape(len(vals), targets.shape[1], len(ALPHAS))
    validation_predictions = (z[VAL] @ coef.reshape(len(vals), -1)).reshape(800, targets.shape[1], len(ALPHAS)) + mean_y[None, :, None]
    losses = ((validation_predictions - targets[VAL, :, None])**2).mean(axis=0)
    best = np.array([min(range(len(ALPHAS)), key=lambda j: (losses[i, j], -ALPHAS[j]))
                     for i in range(targets.shape[1])])
    selected = coef[:, np.arange(targets.shape[1]), best]
    prediction = z @ selected + mean_y
    return prediction, {"selected_alpha": ALPHAS[best].tolist(),
        "validation_mse_by_alpha": losses.tolist(),
        "fitted_coefficients": x.shape[1] + 1}


def make_dataset(replicate, slot):
    rng = np.random.default_rng(310000000 + replicate*100 + slot)
    x = correlated_pairs(rng, 388)
    h = correlated_pairs(rng, 36)
    noise = correlated_pairs(rng, 1)[:, 0]
    current_oracle = 1 + np.sqrt(.5)*x[:, 0]
    history_oracle = current_oracle[:, None] + h[:, :1]*np.sqrt(DELTAS)[None, :]
    target = history_oracle + noise[:, None]*np.sqrt(.5-DELTAS)[None, :]
    assert target.shape == (6000, 4)
    assert np.all(target[TRAIN].mean(axis=0) >= .02) and np.all(target[TRAIN].std(axis=0, ddof=1) >= .02)
    feature_rng = np.random.default_rng(320000000 + replicate*100 + slot)
    w = feature_rng.normal(size=(388, 36)) / np.sqrt(388)
    bias = feature_rng.uniform(0, 2*np.pi, size=36)
    rff = np.sqrt(2/36)*np.cos(standardize(x) @ w + bias)
    matrices = {"current_linear": x, "current_rff": np.concatenate([x, rff], axis=1),
                "history": np.concatenate([x, h], axis=1)}
    for arm in ["current_rff", "history"]:
        assert matrices[arm].shape[1] == 424
        np.testing.assert_array_equal(matrices[arm][:, :388], x)
    checks = {"replicate": replicate, "slot": slot,
        "x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "h_sha256": hashlib.sha256(h.tobytes()).hexdigest(),
        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "test_target_mean": target[TEST].mean(axis=0).tolist(),
        "test_target_variance": target[TEST].var(axis=0).tolist(),
        "test_x_h_correlation": float(np.corrcoef(x[TEST, 0], h[TEST, 0])[0, 1])}
    return matrices, target, current_oracle, history_oracle, checks


def evaluate_dataset(replicate, slot):
    matrices, target, oracle_c, oracle_h, checks = make_dataset(replicate, slot)
    all_predictions, tuning = [], []
    for arm in ARMS:
        p, details = ridge_targets(matrices[arm], target)
        all_predictions.append(p[TEST])
        tuning.append({"replicate": replicate, "slot": slot, "arm": arm, **details})
    pred = np.stack(all_predictions, axis=-1)  # example, delta, arm
    y = target[TEST]
    sst = ((y-y.mean(axis=0))**2).sum(axis=0)
    assert (sst > 0).all() and np.isfinite(pred).all()
    mse = ((pred-y[:, :, None])**2).mean(axis=0)
    mae = np.abs(pred-y[:, :, None]).mean(axis=0)
    scores = 1 - mse*2000/sst[:, None]
    oracle_c_score = 1 - ((y-oracle_c[TEST, None])**2).sum(axis=0)/sst
    oracle_h_score = 1 - ((y-oracle_h[TEST])**2).sum(axis=0)/sst
    deltas = scores[:, 2:3] - scores[:, :2]
    mae_advantage = mae[:, :2] - mae[:, 2:3]
    # Paired family bootstrap, common to all scenarios and comparisons.
    rng = np.random.default_rng(330000000 + replicate*100 + slot)
    counts = rng.multinomial(1000, np.full(1000, .001), size=1000).astype(np.float64)
    ys = y.reshape(1000, 2, 4).sum(axis=1)
    y2s = (y*y).reshape(1000, 2, 4).sum(axis=1)
    boot_sst = counts @ y2s - (counts @ ys)**2/2000
    numerator = ((pred[:, :, :2]-y[:, :, None])**2 - (pred[:, :, 2:3]-y[:, :, None])**2)
    family_numerator = numerator.reshape(1000, 2, 4, 2).sum(axis=1)
    boot_num = (counts @ family_numerator.reshape(1000, 8)).reshape(1000, 4, 2)
    boot_delta = boot_num/boot_sst[:, :, None]
    np.testing.assert_allclose(family_numerator.sum(axis=0)/sst[:, None], deltas, rtol=1e-10, atol=1e-12)
    rows = []
    for di, delta in enumerate(DELTAS):
        rows.append({"replicate": replicate, "slot": slot, "planted_delta": float(delta),
            "current_linear_r2": float(scores[di, 0]), "current_rff_r2": float(scores[di, 1]),
            "history_r2": float(scores[di, 2]),
            "delta_vs_linear": float(deltas[di, 0]), "delta_vs_rff": float(deltas[di, 1]),
            "mae_advantage_vs_linear": float(mae_advantage[di, 0]), "mae_advantage_vs_rff": float(mae_advantage[di, 1]),
            "oracle_current_r2": float(oracle_c_score[di]), "oracle_history_r2": float(oracle_h_score[di]),
            "realized_oracle_delta": float(oracle_h_score[di]-oracle_c_score[di])})
    return deltas, mae_advantage, boot_delta, rows, tuning, checks


def wilson(successes, n):
    z = 1.959963984540054
    p = successes/n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def run(out):
    started = time.time()
    spec = json.loads((ROOT/"CALIBRATION_PROTOCOL.json").read_text())
    manifest = json.loads((ROOT/"FROZEN_INPUT_SHA256.json").read_text())
    for path, digest in manifest.items():
        assert sha(ROOT/path) == digest, path
    assert spec["replicates"] == 24 and spec["datasets_per_replicate"] == 6
    assert spec["planted_population_delta_r2"] == DELTAS.tolist()
    assert not out.exists(), "Use a new output directory"
    out.mkdir(parents=True)
    (out/"bootstrap_panels").mkdir()
    dataset_rows, panel_rows, selections, generation = [], [], [], []
    for replicate in range(24):
        effects, absolute, draws = [], [], []
        for slot in range(6):
            delta, mae, bootstrap, rows, tuning, checks = evaluate_dataset(replicate, slot)
            effects.append(delta); absolute.append(mae); draws.append(bootstrap)
            dataset_rows.extend(rows); selections.extend(tuning); generation.append(checks)
        effects = np.asarray(effects)
        point = effects.mean(axis=0)
        mae = np.asarray(absolute).mean(axis=0)
        boot = np.asarray(draws).mean(axis=0)
        np.savez_compressed(out/"bootstrap_panels"/f"replicate_{replicate:02d}.npz", delta_r2=boot)
        intervals = np.quantile(boot, [.025, .975], axis=0)
        passes = []
        for di, delta in enumerate(DELTAS):
            row = {"replicate": replicate, "planted_delta": float(delta)}
            comparator_pass = []
            for bi, baseline in enumerate(["linear", "rff"]):
                criteria = {"effect_threshold": bool(point[di, bi] >= .01),
                    "ci_above_zero": bool(intervals[0, di, bi] > 0),
                    "all_six_positive": bool(np.all(effects[:, di, bi] > 0)),
                    "mae_not_worse": bool(mae[di, bi] >= 0)}
                passed = all(criteria.values())
                comparator_pass.append(passed)
                row.update({f"mean_delta_vs_{baseline}": float(point[di, bi]),
                    f"ci_low_vs_{baseline}": float(intervals[0, di, bi]),
                    f"ci_high_vs_{baseline}": float(intervals[1, di, bi]),
                    f"mae_advantage_vs_{baseline}": float(mae[di, bi]),
                    f"positive_datasets_vs_{baseline}": int(np.sum(effects[:, di, bi] > 0)),
                    **{f"{key}_vs_{baseline}": value for key, value in criteria.items()}})
            row["screen_passed"] = all(comparator_pass)
            passes.append(int(row["screen_passed"]))
            panel_rows.append(row)
        pd.DataFrame(dataset_rows).to_csv(out/"dataset_metrics.csv", index=False)
        pd.DataFrame(panel_rows).to_csv(out/"panel_metrics.csv", index=False)
        print(f"PANEL {replicate+1:02d}/24 completed; passes at delta [0,.01,.05,.10]: {passes}", flush=True)
    panels = pd.DataFrame(panel_rows)
    datasets = pd.DataFrame(dataset_rows)
    summary = []
    for delta, group in panels.groupby("planted_delta"):
        successes = int(group.screen_passed.sum())
        row = {"planted_population_delta_r2": float(delta), "panels": len(group),
            "detected_panels": successes, "detection_fraction": successes/len(group),
            "monte_carlo_wilson_ci95": wilson(successes, len(group)),
            "mean_observed_delta_vs_rff": float(group.mean_delta_vs_rff.mean()),
            "mean_observed_delta_vs_linear": float(group.mean_delta_vs_linear.mean()),
            "mean_realized_oracle_delta": float(datasets.loc[datasets.planted_delta == delta, "realized_oracle_delta"].mean())}
        for baseline in ["linear", "rff"]:
            for criterion in ["effect_threshold", "ci_above_zero", "all_six_positive", "mae_not_worse"]:
                row[f"panels_passing_{criterion}_vs_{baseline}"] = int(group[f"{criterion}_vs_{baseline}"].sum())
        summary.append(row)
    for path, digest in manifest.items():
        assert sha(ROOT/path) == digest, path
    dump(out/"detection_summary.json", summary)
    dump(out/"validation_choices.json", selections)
    dump(out/"generation_checks.json", generation)
    assert len(dataset_rows) == 576 and len(panel_rows) == 96 and len(selections) == 432
    result = {"study_id": "T2-C1", "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time()-started, "independent_panel_replicates": 24,
        "unique_synthetic_datasets": 144, "synthetic_examples": 864000,
        "scenario_dataset_evaluations": 576, "ridge_target_fits": 1728, "alpha_candidates": 8640,
        "shared_feature_factorizations": 432,
        "null_control_check_passed": summary[0]["detected_panels"] <= 1,
        "strong_signal_control_check_passed": summary[-1]["detected_panels"] >= 22,
        "calibration_checks_passed": summary[0]["detected_panels"] <= 1 and summary[-1]["detected_panels"] >= 22,
        "natural_history_evidence_added": False, "base_models_loaded": 0, "base_model_training_updates": 0,
        "reserved_model_seeds_opened": [], "prior_test_sets_used_for_fitting": False,
        "protocol_sha256": sha(ROOT/"CALIBRATION_PROTOCOL.json"), "runner_sha256": sha(__file__),
        "frozen_inputs_unchanged": True, "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    dump(out/"calibration_result.json", result)
    print("CALIBRATION COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        run(args.out.resolve())
