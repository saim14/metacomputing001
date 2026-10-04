"""Read-only numerical recheck of completed P1-A0 outputs; no fitting."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

import experiment as ex


def verify(output):
    output = Path(output)
    result = json.loads((output / "result.json").read_text())
    protocol = json.loads((output / "FROZEN_PROTOCOL.json").read_text())
    frozen = json.loads((output / "FREEZE.json").read_text())
    ex.verify_freeze(frozen)
    assert frozen["frozen_at_utc"] < result["training_started_at_utc"] < result["completed_at_utc"]
    assert ex.sha256(output / "FROZEN_PROTOCOL.json") == frozen["files_sha256"]["P1_A0_protocol.json"]
    with zipfile.ZipFile(ex.ROOT / "vendor/T1_source_0.1.0.zip") as original:
        for name in ("tiny_autograd.py", "tiny_transformer.py"):
            assert original.read("src/" + name) == (ex.ROOT / "vendor/t1_original/src" / name).read_bytes()
    with np.load(output / "development_data.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    manifest = json.loads((output / "partitions.json").read_text())
    train_keys = {tuple(sorted(x[1:])) for x in data["train_tokens"]}
    cal_keys = {tuple(sorted(x[1:])) for x in data["calibration_tokens"]}
    assert not train_keys & cal_keys
    assert len(train_keys) == 581 and len(cal_keys) == 193
    assert set(data) == {f"{split}_{key}" for split in ("train", "calibration")
                         for key in ("tokens", "labels", "family_ids")}
    labels, fids = data["calibration_labels"], data["calibration_family_ids"]
    verified = []
    for checkpoint in result["checkpoints"]:
        update = checkpoint["update"]
        with np.load(output / f"calibration_trajectory_{update:04d}.npz", allow_pickle=False) as saved:
            logits, probabilities, states = saved["logits"], saved["probabilities"], saved["states"]
            decisions = np.argmax(logits, axis=-1)
            accuracy = np.mean(decisions == labels[:, None], axis=0)
            np.testing.assert_array_equal(accuracy, checkpoint["calibration"]["accuracy_by_step"])
            assert np.all(np.isfinite(states)) and np.all(np.isfinite(logits))
            np.testing.assert_allclose(probabilities.sum(-1), 1, atol=2e-6)
            for recorded in checkpoint["calibration"]["decision_change"]:
                flips = decisions[:, recorded["step"]] != decisions[:, -1]
                assert int(flips.sum()) == recorded["changed_examples"]
                assert int((~flips).sum()) == recorded["unchanged_examples"]
                assert len(np.unique(fids[flips])) == recorded["changed_families"]
                assert len(np.unique(fids[~flips])) == recorded["unchanged_families"]
            model = ex.ArithmeticTransformer(protocol, seed=8003)
            model.load(output / f"model_update_{update:04d}.npz")
            recreated = ex.extract(model, data["calibration_tokens"][:16])
            np.testing.assert_allclose(recreated["states"], states[:16], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(recreated["logits"], logits[:16], atol=2e-5, rtol=2e-5)
        # Training accuracy is recomputed from the saved weights, not the report.
        train_prediction = ex.extract(model, data["train_tokens"])["logits"].argmax(-1)
        train_accuracy = (train_prediction == data["train_labels"][:, None]).mean(0)
        np.testing.assert_array_equal(train_accuracy, checkpoint["train"]["accuracy_by_step"])
        verified.append(update)
    records = [json.loads(line) for line in (output / "training_log.jsonl").read_text().splitlines()]
    assert [x["update"] for x in records] == list(range(1, 1201))
    assert all(np.isfinite(x["loss"]) and np.isfinite(x["grad_norm"]) for x in records)
    assert verified == [0, 100, 400, 1200]
    final = result["checkpoints"][-1]["calibration"]
    competence = final["accuracy_by_step"][-1] >= 0.9
    eligible = []
    for row in final["decision_change"]:
        counts = [row["changed_examples"], row["unchanged_examples"]]
        families = [row["changed_families"], row["unchanged_families"]]
        passed = min(counts) >= 50 and min(families) >= 25 and min(counts)/sum(counts) >= 0.1
        assert passed == row["eligible"]
        if passed:
            eligible.append(row["step"])
    expected = "competence_failed" if not competence else (
        "ready_for_observer_design" if len(eligible) >= 2 else "target_failed")
    assert result["status"] == expected
    assert result["history_probes_fitted"] == result["reserved_examples_evaluated"] == 0
    return {"passed": True, "verified_checkpoints": verified,
            "calibration_accuracy_and_all_flip_counts_recomputed": True,
            "training_accuracy_recomputed_from_all_saved_weights": True,
            "vendor_sources_match_original_archive": True,
            "freeze_preceded_training_and_unchanged": True,
            "reserve_absent_from_evaluated_data": True,
            "status_recomputed": expected,
            "note": "Separate recomputation script, not independent model replication."}


if __name__ == "__main__":
    import sys
    result = verify(sys.argv[1])
    ex.write_json(Path(sys.argv[1]) / "result_verification.json", result)
    print(json.dumps(result, indent=2))
