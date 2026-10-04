"""Numerical and leakage fixtures, never natural observer results."""
import numpy as np

import run_a1 as run


def method_checks(output):
    rng = np.random.default_rng(60013)
    x = rng.normal(size=(170, 11))
    y = rng.normal(size=170)
    fids = np.repeat(np.arange(34), 5)
    w = run.family_weights(fids)
    errors = []
    for penalty, fit in zip(run.PENALTIES, run.ridge_grid(x, y, w, run.PENALTIES)):
        design = np.column_stack([x, np.ones(len(x))])
        penalization = np.column_stack([np.sqrt(penalty) * np.eye(x.shape[1]), np.zeros(x.shape[1])])
        augmented = np.vstack([np.sqrt(w)[:, None] * design, penalization])
        expected = np.linalg.lstsq(augmented, np.concatenate([np.sqrt(w)*y, np.zeros(x.shape[1])]), rcond=None)[0]
        actual = np.r_[fit["coef"], fit["intercept"]]
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        errors.append(float(np.max(np.abs(actual - expected))))
    np.testing.assert_allclose(run.family_weights(np.array([1, 1, 2])), [0.25, 0.25, 0.5])
    m = run.metrics(np.array([0., 1., 1., 0.]), np.array([0.1, 0.9, 0.8, 0.2]), np.arange(4))
    np.testing.assert_allclose(m["brier"], 0.025)
    assert m["roc_auc"] == 1 and m["balanced_accuracy"] == 1
    assert run.metrics(np.array([0., 1.]), np.array([0.5, 0.5]), np.arange(2))["roc_auc"] == 0.5
    tr = {"states": rng.normal(size=(20, 7, 4, 24)),
          "probabilities": np.ones((20, 7, 17))/17,
          "logits": rng.normal(size=(20, 7, 17))}
    for step in run.STEPS:
        before = run.features(tr, step)
        altered = {k: v.copy() for k, v in tr.items()}
        altered["states"][:, step+1:] = 1e9
        altered["probabilities"][:, step+1:] = -999
        altered["logits"][:] = 1e9
        after = run.features(altered, step)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
        assert before[0].shape == (20, 115) and before[1].shape == (20, step * 115)
    fids = np.repeat(np.arange(90), 3)
    c, h = rng.normal(size=(len(fids), 115)), rng.normal(size=(len(fids), 460))
    donor = run.donor_map(fids, rng)
    assert np.all(fids[donor] != fids)
    assert np.array_equal(np.sort(donor), np.arange(len(fids)))
    shuffled, record = run.history_control(h, fids, 4, "shuffled_history", (1,))
    for i in range(4):
        expected = h[i].reshape(4, -1)[record["orders"][i]].reshape(-1)
        np.testing.assert_array_equal(shuffled[i], expected)
    for arm in run.ARMS:
        pre = run.fit_preprocessor(c, h, fids, arm, (1, 2))
        np.testing.assert_allclose(pre["c_scale"]["mean"], run.family_weights(fids) @ c)
        test_c, test_h = c[:20] + 123, h[:20] - 73
        original_mean = pre["c_scale"]["mean"].copy()
        transformed = run.transform(test_c, test_h, pre)
        np.testing.assert_array_equal(original_mean, pre["c_scale"]["mean"])
        assert transformed.shape[1] == (115 if arm == "current" else 575)
        assert np.all(np.isfinite(transformed))
        path = output / f"method_preprocessor_{arm}.npz"
        run.save_tree(path, pre)
        np.testing.assert_array_equal(run.transform(test_c, test_h, run.load_tree(path)), transformed)
    z = rng.normal(size=(8, 5))
    kernel = run.gaussian(z, z, 0.5)
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-12)
    np.testing.assert_allclose(kernel, kernel.T)
    expected_train = np.repeat(np.arange(581), 3)
    for seed in run.SPLITS:
        split = run.observer_split(expected_train, seed)
        a, b = expected_train[split["fit"]], expected_train[split["tune"]]
        assert len(np.unique(a)) == 465 and len(np.unique(b)) == 116
        assert not set(a) & set(b)
        assert len(a) + len(b) == len(expected_train)
    fit_idx, tune_idx = np.arange(210), np.arange(210,270)
    toy_y = (c[:,0] > 0).astype(float)
    toy_data = {part: {"c":c[idx], "h":h[idx], "fids":fids[idx], "y":toy_y[idx]}
                for part,idx in [("fit",fit_idx),("tune",tune_idx)]}
    pipe, record = run.fit_arm(toy_data,"current",4,("method",))
    expected_scores = [run.metrics(toy_y[tune_idx], p, fids[tune_idx])["brier"]
                       for p in record["tuning_predictions"]]
    np.testing.assert_allclose(expected_scores,[r["tune_brier"] for r in record["candidates"]])
    assert record["selected_index"] == int(np.argmin(expected_scores))
    path = output / "method_selected_pipeline.npz"
    run.save_tree(path, pipe)
    np.testing.assert_array_equal(run.predict(pipe,c[tune_idx],h[tune_idx]),
                                  run.predict(run.load_tree(path),c[tune_idx],h[tune_idx]))
    return {"passed": True, "ridge_against_augmented_least_squares_max_error": max(errors),
            "family_weights_and_brier_auc_checked": True, "future_and_target_feature_exclusion": True,
            "all_token_dimensions_checked": True, "donor_bijection_and_family_exclusion": True,
            "shuffle_slots_checked": True, "fit_only_scaling_and_transform_immutability": True,
            "all_six_preprocessor_serializations_exact": True, "gaussian_kernel_fixture": True,
            "all_family_splits_disjoint": True, "natural_target_fits": 0,
            "tuning_selection_and_selected_pipeline_roundtrip": True,
            "method_target_ridge_solves": 12, "method_label_free_residualization_solves": 1}
