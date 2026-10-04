"""Recompute saved results and decisions without training or observer fitting."""
import json
from pathlib import Path

import numpy as np

import run_a0b as run


def verify(folder):
    folder = Path(folder)
    result = json.loads((folder / "result.json").read_text())
    p = json.loads((folder / "FROZEN_PROTOCOL.json").read_text())
    frozen = json.loads((folder / "FREEZE.json").read_text())
    run.check_freeze(frozen)
    assert p == run.read_protocol()
    old = run.validate_protocol(p)
    with np.load(folder / "development_data.npz", allow_pickle=False) as z:
        data = {name: z[name] for name in z.files}
    expected, manifest = run.base.make_partitions(old)
    for split, values in expected.items():
        for key, value in values.items():
            np.testing.assert_array_equal(value, data[f"{split}_{key}"])
    assert set(data) == {f"{s}_{k}" for s in expected for k in expected[s]}
    assert manifest == json.loads((folder / "partitions.json").read_text())
    assert not set(expected["train"]["family_ids"]) & set(expected["calibration"]["family_ids"])
    assert [[r["seed"],r["arm"]] for r in result["runs"]] == p["design"]["run_order"]
    verified_runs, max_error = [], 0.0
    first_step_logs = {}
    for recorded in result["runs"]:
        seed, arm = recorded["seed"], recorded["arm"]
        sub = folder / f"seed_{seed}_{arm}"
        assert frozen["frozen_at_utc"] < recorded["started_at_utc"] <= recorded["training_started_at_utc"] < recorded["completed_at_utc"]
        with np.load(folder / f"batch_stream_{seed}.npz", allow_pickle=False) as z:
            batches = z["indices"]
        assert batches.shape == (6000,64) and batches.dtype == np.int64
        regenerated = np.random.default_rng(500000+seed).integers(0,2992,size=(6000,64),dtype=np.int64)
        np.testing.assert_array_equal(regenerated,batches)
        assert recorded["batch_index_sha256"] == run.array_digest(batches)
        log = [json.loads(line) for line in (sub / "training_log.jsonl").read_text().splitlines()]
        assert [r["update"] for r in log] == list(range(1,6001))
        assert all(np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"]) for r in log)
        first_step_logs[(seed,arm)] = (log[0]["loss"], log[0]["grad_norm"])
        assert [c["update"] for c in recorded["checkpoints"]] == [0,1200,3000,6000]
        for checkpoint in recorded["checkpoints"]:
            update = checkpoint["update"]
            model = run.base.ArithmeticTransformer(old, seed=seed)
            model.load(sub / f"model_{update:04d}.npz")
            if update == 0:
                assert run.weight_digest(model) == recorded["initial_weight_sha256"]
            if update:
                with np.load(sub / f"optimizer_{update:04d}.npz",allow_pickle=False) as z:
                    assert int(z["step_count"]) == update
                    assert len(z.files) == 2*len(model.parameters())+1
                    for i,parameter in enumerate(model.parameters()):
                        for prefix in ("m","v"):
                            assert z[f"{prefix}_{i}"].shape == parameter.shape
                            assert np.all(np.isfinite(z[f"{prefix}_{i}"]))
                        assert np.all(z[f"v_{i}"] >= 0)
            for split in ("train","calibration"):
                labels, fids = data[f"{split}_labels"], data[f"{split}_family_ids"]
                with np.load(sub / f"{split}_predictions_{update:04d}.npz",allow_pickle=False) as z:
                    logits,probabilities = z["logits"],z["probabilities"]
                # Full forward recomputation at every saved checkpoint, no fitting.
                recreated = run.base.extract(model,data[f"{split}_tokens"])
                error = float(np.max(np.abs(recreated["logits"]-logits)))
                max_error = max(max_error,error)
                np.testing.assert_allclose(recreated["logits"],logits,atol=2e-5,rtol=2e-5)
                np.testing.assert_allclose(probabilities,run.base.numpy_softmax(logits),atol=2e-6,rtol=2e-6)
                predictions = logits.argmax(-1)
                accuracies = (predictions == labels[:,None]).mean(0)
                np.testing.assert_array_equal(accuracies,checkpoint[split]["accuracy_by_step"])
                for row in checkpoint[split]["decision_change"]:
                    changed = predictions[:,row["step"]] != predictions[:,-1]
                    counts = [int((~changed).sum()),int(changed.sum())]
                    families = [len(np.unique(fids[~changed])),len(np.unique(fids[changed]))]
                    assert counts == [row["unchanged_examples"],row["changed_examples"]]
                    assert families == [row["unchanged_families"],row["changed_families"]]
                    gate = min(counts)>=50 and min(families)>=25 and min(counts)/sum(counts)>=0.1
                    assert gate == row["eligible"]
                if split == "calibration":
                    errors = run.base.resume_audit(model,recreated)
                    for e in errors.values():
                        assert e <= 2e-5 + 2e-5*float(np.max(np.abs(logits)))
            assert max(checkpoint["resume_errors"].values()) <= 2e-5 + 2e-5*float(np.max(np.abs(logits)))
        verified_runs.append([seed,arm])
    for seed in (2017,2027):
        assert first_step_logs[(seed,"no_decay")] == first_step_logs[(seed,"matrix_decay")]
    # Independent decision reconstruction; do not call the production decide().
    ready, recomputed_regimes = [], []
    for name in ("no_decay","matrix_decay"):
        group = [r for r in result["runs"] if r["arm"]==name]
        competent = all(r["checkpoints"][-1]["calibration"]["accuracy_by_step"][-1]>=0.9 for r in group)
        common = set([2,3,4,5])
        for r in group:
            rows = r["checkpoints"][-1]["calibration"]["decision_change"]
            common &= {row["step"] for row in rows if row["eligible"]}
        passed = competent and len(common)>=2
        recorded = next(x for x in result["decision"]["regimens"] if x["arm"]==name)
        assert recorded["competence_passed_all_seeds"] == competent
        assert recorded["common_eligible_steps"] == sorted(common)
        assert recorded["ready_for_observer_design"] == passed
        if passed:
            ready.append(name)
        recomputed_regimes.append({"arm":name,"ready":passed})
    assert result["decision"]["selected_development_candidate"] == (ready[0] if ready else None)
    differences = []
    for row in result["decision"]["paired_effects"]:
        subset = {r["arm"]:r for r in result["runs"] if r["seed"]==row["seed"]}
        expected_diff = subset["matrix_decay"]["final_calibration_accuracy"] - subset["no_decay"]["final_calibration_accuracy"]
        assert expected_diff == row["decay_minus_no_decay_accuracy"]
        differences.append(expected_diff)
    assert float(np.mean(differences)) == result["decision"]["mean_paired_accuracy_difference"]
    assert result["total_updates"] == 24000
    assert result["reserved_examples_evaluated"] == result["decision"]["history_probes_fitted"] == 0
    run.check_freeze(frozen)
    return {"passed":True,"verified_runs":verified_runs,"saved_checkpoints_verified":16,
            "all_train_and_calibration_logits_recomputed":True,"max_logit_error":max_error,
            "all_accuracy_flip_counts_and_gates_recomputed":True,"paired_initializations_streams_and_first_gradient_verified":True,
            "all_24000_updates_and_optimizer_moments_verified":True,"reserved_examples_absent":True,
            "scientific_freeze_preceded_all_training_and_is_unchanged":True,
            "regimen_decisions_recomputed":recomputed_regimes,
            "scope":"Separate numerical audit; not independent model replication."}


if __name__ == "__main__":
    import sys
    folder = Path(sys.argv[1])
    result = verify(folder)
    run.write_json(folder / "result_verification.json",result)
    print(json.dumps(result,indent=2))
