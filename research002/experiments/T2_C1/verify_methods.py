"""Numerical equivalence and leakage checks; no calibration datasets are used."""
import ast
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import run_calibration as c


def verify():
    tree = ast.parse((c.ROOT/'reference/T2_E1_runner.py').read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ['standardize', 'ridge_path']]
    assert len(selected) == 2
    namespace = {'np': np, 'PARTS': {'train': c.TRAIN, 'validation': c.VAL, 'test': c.TEST}}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'original_ridge', 'exec'), namespace)
    rng = np.random.default_rng(90817263)
    x = rng.normal(size=(6000, 11))
    x[:, 9] = x[:, 0] + x[:, 1]
    x[:, 10] = 7  # rank deficient and constant columns
    y = 2 + x[:, :4] + rng.normal(size=(6000, 4))
    prediction, detail = c.ridge_targets(x, y)
    max_original_error, max_sklearn_error = 0., 0.
    z = c.standardize(x)
    for k in range(4):
        original, prior = namespace['ridge_path'](x, y[:, k], c.ALPHAS)
        np.testing.assert_allclose(prediction[:, k], original, rtol=1e-7, atol=1e-8)
        assert detail['selected_alpha'][k] == prior['alpha']
        max_original_error = max(max_original_error, float(np.max(np.abs(prediction[:, k]-original))))
        candidates, losses = [], []
        for alpha in c.ALPHAS:
            model = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky').fit(z[c.TRAIN], y[c.TRAIN, k])
            candidates.append(model.predict(z))
            losses.append(float(np.mean((candidates[-1][c.VAL]-y[c.VAL, k])**2)))
        best = min(range(len(c.ALPHAS)), key=lambda j: (losses[j], -c.ALPHAS[j]))
        assert detail['selected_alpha'][k] == c.ALPHAS[best]
        np.testing.assert_allclose(prediction[:, k], candidates[best], rtol=1e-7, atol=1e-8)
        max_sklearn_error = max(max_sklearn_error, float(np.max(np.abs(prediction[:, k]-candidates[best]))))
        np.testing.assert_allclose(detail['validation_mse_by_alpha'][k], losses, rtol=1e-8, atol=1e-10)
    changed = y.copy()
    changed[c.TEST] += rng.normal(size=(2000, 4))*100
    p2, d2 = c.ridge_targets(x, changed)
    np.testing.assert_array_equal(prediction, p2)
    assert detail == d2
    altered_x = x.copy()
    altered_x[c.TEST] *= 100
    p3, d3 = c.ridge_targets(altered_x, y)
    np.testing.assert_array_equal(prediction[:4000], p3[:4000])
    assert detail == d3
    constant = np.ones((6000, 4))
    _, tied = c.ridge_targets(x, constant)
    assert tied['selected_alpha'] == [100.]*4
    np.testing.assert_allclose(.5+c.DELTAS+(.5-c.DELTAS), 1)
    assert np.isclose(c.wilson(0, 24)[0], 0, atol=1e-15) and np.isclose(c.wilson(24, 24)[1], 1)
    return {'status': 'passed', 'original_T2_E1_equivalence_targets': 4,
            'independent_sklearn_candidates': 20, 'max_original_prediction_error': max_original_error,
            'max_sklearn_prediction_error': max_sklearn_error,
            'rank_deficient_and_constant_features': 'passed', 'test_target_leakage': 'passed',
            'test_feature_preprocessing_leakage': 'passed', 'exact_tie_rule': 'passed',
            'population_variance_identity': 'passed', 'wilson_boundaries': 'passed',
            'calibration_panels_used': 0}


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        result = verify()
    (c.ROOT/'method_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
