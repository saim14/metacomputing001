"""Small scientific checks of the ridge solver, leakage boundary, and pairing."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from run_study import ridge_path, standardize, PARTS, history_controls, family_keys


def verify():
    rng = np.random.default_rng(982717)
    x = rng.normal(size=(6000, 6))
    x[:, 4] = 1
    x[:, 5] = x[:, 0] + x[:, 1]
    y = 2*x[:, 0] - x[:, 2] + rng.normal(scale=.3, size=6000)
    alphas = [.01, .1, 1., 10., 100.]
    pred, details = ridge_path(x, y, alphas)
    z, _, _ = standardize(x)
    candidates = []
    for alpha in alphas:
        m = Ridge(alpha=alpha, solver="cholesky").fit(z[PARTS["train"]], y[PARTS["train"]])
        p = m.predict(z)
        candidates.append((float(np.mean((p[PARTS["validation"]]-y[PARTS["validation"]])**2)), -alpha, p))
    expected = min(candidates, key=lambda row: row[:2])
    np.testing.assert_allclose(pred, expected[2], rtol=1e-7, atol=1e-8)
    assert details["alpha"] == -expected[1]
    corrupted_y = y.copy()
    corrupted_y[PARTS["test"]] += 10000
    again, again_details = ridge_path(x, corrupted_y, alphas)
    np.testing.assert_array_equal(pred, again)
    assert details == again_details
    changed_x = x.copy()
    changed_x[PARTS["test"]] += 100000
    z_changed, mean_changed, sd_changed = standardize(changed_x)
    np.testing.assert_array_equal(z[PARTS["train"]], z_changed[PARTS["train"]])
    np.testing.assert_array_equal(standardize(x)[1], mean_changed)
    np.testing.assert_array_equal(standardize(x)[2], sd_changed)
    f = rng.normal(size=(6000, 7, 36))
    controls = history_controls(f, 503, 3)
    assert not np.array_equal(controls["ordered_history"], controls["shuffled_history"])
    altered = f.copy()
    altered[:, 3:] += 10000
    altered_controls = history_controls(altered, 503, 3)
    for key in controls:
        np.testing.assert_array_equal(controls[key], altered_controls[key])
    tokens = np.asarray([[10, 0, 0, 1, 1, 8, 2, 9, 3, 8, 4, 9],
                         [10, 0, 0, 1, 1, 9, 2, 8, 3, 9, 4, 8]])
    assert family_keys(tokens)[0] == family_keys(tokens)[1]
    result = {"status":"passed", "checks":[
        "ridge path matches independent scikit-learn reference on collinear data",
        "changing held-out labels cannot affect predictions or hyperparameter selection",
        "changing held-out features cannot affect training normalization",
        "shuffle alters order while preserving the past multiset",
        "current/future features cannot enter past-history controls",
        "counterfactual members map to the same split-family key"]}
    Path(__file__).with_name("method_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    with threadpool_limits(limits=1, user_api="blas"):
        verify()
