"""Recompute every reported held-out score from saved predictions; no refitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from run_study import ARMS, OBSERVERS, STEPS, sha

ROOT = Path(__file__).resolve().parent
out = ROOT / "results"
metrics = pd.read_csv(out / "probe_metrics.csv")
records = json.loads((out / "seed_comparisons.json").read_text())
panel = json.loads((out / "panel_comparisons.json").read_text())
scores_checked = 0
for seed in [503, 607, 709, 811, 907, 1009]:
    with np.load(out / "predictions" / f"seed_{seed}.npz", allow_pickle=False) as z:
        y, pred, eligible = z["target"], z["prediction"], z["eligible"]
    assert np.isfinite(y).all() and np.isfinite(pred).all()
    for si, step in enumerate(STEPS):
        for oi, observer in enumerate(OBSERVERS):
            for ai, arm in enumerate(ARMS):
                row = metrics.query("seed == @seed and step == @step and observer == @observer and arm == @arm").iloc[0]
                np.testing.assert_allclose([r2_score(y[:, si], pred[:, si, oi, ai]),
                    mean_absolute_error(y[:, si], pred[:, si, oi, ai]),
                    mean_squared_error(y[:, si], pred[:, si, oi, ai])],
                    row[["test_r2", "test_mae", "test_mse"]].astype(float), rtol=1e-10, atol=1e-10)
                scores_checked += 1
    for row in [r for r in records if r["seed"] == seed and r["delta_r2"] is not None]:
        sub = metrics[(metrics.seed == seed) & (metrics.observer == row["observer"])
                      & metrics.step.isin(row["eligible_steps"])]
        candidate = sub[sub.arm == "ordered_history"].set_index("step")
        baseline = sub[sub.arm == row["baseline"]].set_index("step")
        np.testing.assert_allclose((candidate.test_r2-baseline.test_r2).mean(), row["delta_r2"], atol=1e-12)
        np.testing.assert_allclose((baseline.test_mae-candidate.test_mae).mean(), row["mae_advantage"], atol=1e-12)
for row in panel:
    relevant = [r for r in records if r["observer"] == row["observer"] and r["baseline"] == row["baseline"] and r["delta_r2"] is not None]
    np.testing.assert_allclose(np.mean([r["delta_r2"] for r in relevant]), row["mean_delta_r2"], atol=1e-12)
    np.testing.assert_allclose(np.mean([r["mae_advantage"] for r in relevant]), row["mean_mae_advantage"], atol=1e-12)
result = json.loads((out/"study_result.json").read_text())
assert result["runner_sha256"] == sha(ROOT/"run_study.py")
assert result["protocol_sha256"] == sha(ROOT/"STUDY_PROTOCOL.json")
check = {"status":"passed", "held_out_cells_recomputed_with_sklearn":scores_checked,
         "seed_and_panel_aggregation_verified":True, "runner_and_protocol_hashes_verified":True,
         "probe_refits":0}
(out/"result_verification.json").write_text(json.dumps(check,indent=2)+"\n")
print(json.dumps(check))
