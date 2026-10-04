"""Post-run audit of immutable evidence; no observer fitting or network training."""
import json
from pathlib import Path
import sys

import numpy as np

import run_a1 as run


def close(a, b, tolerance=1e-11):
    np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


def independent_family_average(values, fids):
    return np.array([values[fids == family].mean() for family in np.unique(fids)])


def interval(values, draws):
    bootstrap_means = np.mean(values[draws], axis=1)
    return {"mean": float(np.mean(values)), "ci95": np.percentile(bootstrap_means, [2.5, 97.5]).tolist(),
            "lower95": float(np.percentile(bootstrap_means, 5)),
            "upper95": float(np.percentile(bootstrap_means, 95))}


def assert_interval(a, b):
    for k in ["mean", "ci95", "lower95", "upper95"]:
        close(a[k], b[k])


def audit(output):
    result = json.loads((output / "result.json").read_text())
    freeze = json.loads((output / "FREEZE.json").read_text())
    started = json.loads((output / "FITTING_STARTED.json").read_text())
    assert freeze["frozen_at_utc"] < started["started_at_utc"] < result["completed_at_utc"]
    assert freeze["files_sha256"] == run.scientific_files()
    assert json.loads((output / "method_checks.json").read_text())["passed"]
    recovery_verified=False
    recovery_path=output/"recovery/RECOVERY.json"
    if recovery_path.exists():
        recovery=json.loads(recovery_path.read_text())
        original=output/recovery["preserved_original"]
        reconstructed=output/recovery["artifact"]
        assert run.digest(original)==recovery["original_sha256"]
        assert run.digest(reconstructed)==recovery["reconstructed_sha256"]
        assert reconstructed.read_bytes()[:original.stat().st_size]==original.read_bytes()
        assert recovery["observer_fits_added"]==recovery["network_training_updates_added"]==0
        recovery_verified=True
    for name, record in json.loads((run.ROOT / "INPUT_PROVENANCE.json").read_text()).items():
        assert run.digest(run.ROOT / name) == record["sha256"]
    stopped = run.ROOT / "stopped_v1"
    for name, expected in json.loads((stopped / "CONTENTS_SHA256.json").read_text()).items():
        assert run.digest(stopped / name) == expected
    old_freeze = json.loads((stopped / "panel_run/FREEZE.json").read_text())["files_sha256"]
    for name, expected in old_freeze.items():
        path = run.ROOT / name if name.startswith("inputs/") else stopped / name
        assert run.digest(path) == expected
    with np.load(run.ROOT / "inputs/development_data.npz", allow_pickle=False) as z:
        development = {k: z[k] for k in z.files}
    expected, partition_manifest = run.base.make_partitions()
    assert set(development) == {f"{s}_{k}" for s in expected for k in expected[s]}
    for part, arrays in expected.items():
        for name, value in arrays.items():
            np.testing.assert_array_equal(value, development[f"{part}_{name}"])
    train_families = set(development["train_family_ids"])
    eval_families = set(development["calibration_family_ids"])
    reserve = set(partition_manifest["family_ids"]["reserve"])
    assert not train_families & eval_families and not reserve & (train_families | eval_families)
    assert len(eval_families) == 193 and len(reserve) == 195
    trajectories, trajectory_error = {}, 0.0
    for seed in run.MODELS:
        model = run.base.ArithmeticTransformer(seed=seed)
        model.load(run.ROOT / f"inputs/model_{seed}.npz")
        for part in ("train", "calibration"):
            with np.load(output / f"trajectory_{seed}_{part}.npz", allow_pickle=False) as z:
                tr = {k: z[k] for k in z.files}
            recomputed = run.base.extract(model, development[f"{part}_tokens"])
            for name in tr:
                error = float(np.max(np.abs(tr[name]-recomputed[name])))
                trajectory_error = max(trajectory_error, error)
                close(tr[name], recomputed[name], 2e-5)
            trajectories[(seed, part)] = tr
    for split_seed in run.SPLITS:
        idx = run.observer_split(development["train_family_ids"], split_seed)
        ffit = set(development["train_family_ids"][idx["fit"]])
        ftune = set(development["train_family_ids"][idx["tune"]])
        assert len(ffit) == 465 and len(ftune) == 116 and not ffit & ftune
    completed = json.loads((output / "completed_cells.json").read_text())
    assert len(completed) == 24
    assert sum(r["target_fits"] for r in completed) == 1080
    observed_stats, vectors = {}, {}
    draws = np.random.default_rng(73001).integers(0, 193, size=(5000,193))
    candidate_count, pipeline_count = 0, 0
    max_prediction_error, max_optimality_residual = 0.0, 0.0
    for item in completed:
        directory = output / item["directory"]
        row = json.loads((directory / "result.json").read_text())
        assert started["started_at_utc"] <= row["started_at_utc"] < row["completed_at_utc"]
        key = tuple(row["key"])
        if key[0] == "synthetic":
            data = run.synthetic_data(development, key[1], key[2])
            arms = run.SYNTHETIC_ARMS
        else:
            data = run.make_natural_data(trajectories, development, *key)
            arms = run.ARMS
            for part in ("fit", "tune"):
                y, f = data[part]["y"], data[part]["fids"]
                for value in (0,1):
                    assert sum(y==value) >= (50 if part=="fit" else 20)
                    assert len(np.unique(f[y==value])) >= (25 if part=="fit" else 10)
        with np.load(directory / "predictions.npz", allow_pickle=False) as z:
            saved = {k:z[k] for k in z.files}
        with np.load(directory / "tuning_predictions.npz", allow_pickle=False) as z:
            tuned = {k:z[k] for k in z.files}
        for saved_arrays, part in [(saved,"eval"),(tuned,"tune")]:
            np.testing.assert_array_equal(saved_arrays["fids"],data[part]["fids"])
            np.testing.assert_array_equal(saved_arrays["y"],data[part]["y"])
        controls = run.load_tree(directory / "control_maps.npz")
        losses = {}
        for arm in arms:
            pipe = run.load_tree(directory / f"pipeline_{arm}.npz")
            arm_data = {}
            for part, values in data.items():
                h = values["h"]
                if arm == "mismatched_history":
                    donor = controls[arm][part]["donor"]
                    assert np.array_equal(np.sort(donor), np.arange(len(h)))
                    assert np.all(values["fids"][donor] != values["fids"])
                    h = h[donor]
                elif arm == "shuffled_history":
                    orders = controls[arm][part]["orders"]
                    assert np.all(np.sort(orders,axis=1) == np.arange(row["step"]))
                    blocks = h.reshape(len(h),row["step"],-1)
                    h = blocks[np.arange(len(h))[:,None],orders].reshape(h.shape)
                arm_data[part] = {**values,"h":h}
            fit = arm_data["fit"]
            pre = pipe["preprocessor"]
            weights = np.array([1/(len(np.unique(fit["fids"])) * np.sum(fit["fids"]==f)) for f in fit["fids"]])
            expected_mean = weights @ fit["c"]
            close(expected_mean,pre["c_scale"]["mean"])
            expected_sd = np.sqrt(weights @ ((fit["c"]-expected_mean)**2))
            close(np.where(expected_sd<1e-8,1,expected_sd),pre["c_scale"]["sd"])
            # Stationarity of the selected ridge objective, without refitting it.
            x = run.transform(fit["c"],fit["h"],pre)
            if pipe["kind"] == "rbf":
                x = run.scale(run.gaussian(x,pipe["landmarks"],pipe["bandwidth"]),pipe["phi_scale"])/8
                assert len(pipe["landmark_indices"]) == 64
                assert len(np.unique(fit["fids"][pipe["landmark_indices"]])) == 64
            residual = x @ pipe["coef"] + pipe["intercept"] - fit["y"]
            gradient = x.T @ (weights*residual) + pipe["penalty"]*pipe["coef"]
            stationarity = max(float(np.max(np.abs(gradient))),abs(float(weights @ residual)))
            max_optimality_residual = max(max_optimality_residual,stationarity)
            assert stationarity < 1e-8
            for part, expected_p in [("eval",saved[arm]),("tune",tuned[arm][row["arms"][arm]["selected_index"]])]:
                val = arm_data[part]
                actual = run.predict(pipe,val["c"],val["h"])
                max_prediction_error=max(max_prediction_error,float(np.max(np.abs(actual-expected_p))))
                close(actual,expected_p)
            tune_scores=[]
            assert tuned[arm].shape == (9,len(tuned["y"]))
            for i,p in enumerate(tuned[arm]):
                assert np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1))
                score=float(independent_family_average((p-tuned["y"])**2,tuned["fids"]).mean())
                tune_scores.append(score)
                close(score,row["arms"][arm]["candidates"][i]["tune_brier"])
            assert row["arms"][arm]["selected_index"] == int(np.argmin(tune_scores))
            losses[arm]=independent_family_average((saved[arm]-saved["y"])**2,saved["fids"])
            close(losses[arm].mean(),row["arms"][arm]["evaluation"]["brier"])
            candidate_count+=9
            pipeline_count+=1
        current=min(["current","expanded_current"],key=lambda a:(row["arms"][a]["selected"]["tune_brier"],a!="current"))
        assert current == row["current_reference"]
        baseline=float(independent_family_average(data["fit"]["y"],data["fit"]["fids"]).mean())
        close(saved["baseline"],baseline)
        losses["baseline"]=independent_family_average((saved["baseline"]-saved["y"])**2,saved["fids"])
        delta=losses[current]-losses["ordered_history"]
        correspondence=losses["mismatched_history"]-losses["ordered_history"]
        utility=losses["baseline"]-losses[current]
        skill=1-losses[current].mean()/losses["baseline"].mean()
        observed_stats[key]={"history_advantage":interval(delta,draws),
            "correspondence_advantage":interval(correspondence,draws),
            "baseline_improvement":interval(utility,draws),"current_skill":float(skill),
            "useful":bool(skill>=0.1 and interval(utility,draws)["lower95"]>0)}
        vectors[key]={"delta":delta,"correspondence":correspondence}
        for name in row.get("timing",{}):
            t=row["timing"][name]
            assert len(t["repetitions_seconds"])==15 and min(t["repetitions_seconds"])>0
            close(np.median(t["repetitions_seconds"]),t["median_seconds"])
    assert candidate_count==1080 and pipeline_count==120
    decision=result["decision"]
    for row in decision["natural_cells"]+decision["synthetic_cells"]:
        actual=observed_stats[tuple(row["key"])]
        for label in ["history_advantage","correspondence_advantage","baseline_improvement"]:
            assert_interval(actual[label],row[label])
        close(actual["current_skill"],row["current_skill"])
        assert actual["useful"]==row["useful"]
    pooled={}
    for split in run.SPLITS:
        keys=[(split,seed,step) for seed in run.MODELS for step in run.STEPS]
        pooled[split]={name:interval(np.mean([vectors[k][name] for k in keys],axis=0),draws)
                       for name in ["delta","correspondence"]}
        assert_interval(pooled[split]["delta"],decision["by_observer_split"][str(split)]["history_advantage"])
        assert_interval(pooled[split]["correspondence"],decision["by_observer_split"][str(split)]["correspondence_advantage"])
    by_model={seed:interval(np.mean([vectors[(314159,seed,t)]["delta"] for t in [3,4]],axis=0),draws) for seed in [2017,2027]}
    strong=sum(observed_stats[("synthetic",seed,0.2)]["history_advantage"]["lower95"]>0 and
               observed_stats[("synthetic",seed,0.2)]["history_advantage"]["mean"]>=0.01 for seed in [9101,9102,9103,9104])
    near=sum(observed_stats[("synthetic",seed,0.1)]["history_advantage"]["lower95"]>0 for seed in [9101,9102,9103,9104])
    null=sum(observed_stats[("synthetic",seed,0.0)]["history_advantage"]["lower95"]>0.01 for seed in [9101,9102,9103,9104])
    sensitive=strong==4 and near>=3 and null==0
    assert decision["sensitivity"]["passed"]==sensitive
    history_conditions=[pooled[314159]["delta"]["lower95"]>0.01,
        all(by_model[s]["mean"]>0 for s in [2017,2027]),pooled[314159]["correspondence"]["lower95"]>0,
        all(pooled[s]["delta"]["mean"]>0.01 for s in [314160,314161])]
    sufficiency_conditions=[all(observed_stats[(314159,s,t)]["history_advantage"]["upper95"]<0.01 for s in [2017,2027] for t in [3,4]),
        all(pooled[s]["delta"]["upper95"]<0.01 for s in [314160,314161]),
        all(observed_stats[(314159,s,t)]["useful"] for s in [2017,2027] for t in [3,4]),sensitive]
    expected_status=("development_history_advantage" if all(history_conditions) else
                     "bounded_practical_current_state_sufficiency" if all(sufficiency_conditions) else
                     "inconclusive_at_fixed_budgets")
    assert decision["status"]==expected_status
    assert list(decision["history_checks"].values())==history_conditions
    assert list(decision["sufficiency_checks"].values())==sufficiency_conditions
    assert not decision["confirmation_opened"] and not decision["awareness_tested"]
    assert result["arithmetic_network_training_updates"]==result["reserved_examples_evaluated"]==0
    assert freeze["files_sha256"]==run.scientific_files()
    return {"passed":True,"completed_at_utc":run.now(),"selected_pipelines_audited":pipeline_count,
            "tuning_candidate_scores_checked":candidate_count,
            "all_selected_evaluation_and_tuning_predictions_recomputed":True,
            "max_prediction_error":max_prediction_error,"selected_ridge_stationarity_max_residual":max_optimality_residual,
            "all_natural_trajectories_recomputed_max_error":trajectory_error,
            "all_family_metrics_and_bootstrap_intervals_recomputed":True,
            "decisions_independently_reconstructed":True,"old_stopped_attempt_preserved":True,
            "derived_cache_recovery_verified":recovery_verified,
            "frozen_sources_and_model_inputs_unchanged":True,"reserve_data_absent":True,
            "scope":"Numerical evidence audit, conditional on the same models and development data; no fitting or new confirmation."}


if __name__=="__main__":
    output=Path(sys.argv[1])
    result=audit(output)
    run.write_json(output/"result_verification.json",result)
    print(json.dumps(result,indent=2))
