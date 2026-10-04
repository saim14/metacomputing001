"""P1-A0B fixed development panel; only NumPy plus the original project code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "prior"))
import experiment as base
from src.tiny_autograd import Adam, cross_entropy


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    header = json.dumps([str(value.dtype), value.shape]).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def weight_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def read_protocol():
    return json.loads((ROOT / "P1_A0B_protocol.json").read_text())


class MatrixAdamW(Adam):
    """Exact old Adam at lambda=0; otherwise decay matrices separately."""
    def __init__(self, parameters, *, weight_decay, **kwargs):
        super().__init__(parameters, **kwargs)
        self.weight_decay = float(weight_decay)
        self.decayed = [p for p in self.parameters if p.ndim >= 2]
        if not 0 <= self.learning_rate * self.weight_decay < 1:
            raise ValueError("Unsupported decay factor.")

    def step(self):
        if self.weight_decay:
            factor = 1.0 - self.learning_rate * self.weight_decay
            for p in self.decayed:
                p.data *= factor
        return super().step()


def validate_protocol(p):
    assert p["study_id"] == "P1-A0B" and p["version"] == "1.0.0"
    d = p["design"]
    assert d["model_seeds"] == [2017, 2027]
    assert d["arms"] == [{"name": "no_decay", "weight_decay": 0.0}, {"name": "matrix_decay", "weight_decay": 1.0}]
    assert d["run_order"] == [[s, a["name"]] for s in d["model_seeds"] for a in d["arms"]]
    assert d["updates_per_run"] == 6000 and d["maximum_total_updates"] == 24000
    assert d["maximum_natural_training_runs"] == 4
    assert d["checkpoints"] == [0, 1200, 3000, 6000]
    assert not p["decisions"]["history_probes_allowed"] and not p["decisions"]["confirmation_allowed"]
    old = base.read_protocol()
    assert old["model"]["d_model"] == 24 and old["model"]["n_steps"] == 6
    assert old["data"]["modulus"] == 17
    for key in ("batch_size", "learning_rate", "beta1", "beta2", "epsilon"):
        assert p["preserved"][key] == old["training"][key]
    assert p["preserved"]["gradient_clip"] == old["training"]["max_grad_norm"]
    assert p["measurements"]["competence_threshold"] == old["measurements"]["competence_gate"]
    assert not set(d["model_seeds"]) & set(p["preserved"]["old_reserved_model_seeds"])
    return old


def scientific_files():
    required = ["P1_A0B_protocol.json", "P1_A0B_Protocol.md", "run_a0b.py", "checks_a0b.py",
                "prior/experiment.py", "prior/test_methods.py", "prior/P1_A0_protocol.json",
                "prior/P1_A0_Protocol.md", "prior/P1_A0_Report.md", "prior/vendor/T1_source_0.1.0.zip",
                "prior/vendor/t1_original/src/__init__.py",
                "prior/vendor/t1_original/src/tiny_autograd.py",
                "prior/vendor/t1_original/src/tiny_transformer.py"]
    return {name: digest(ROOT / name) for name in required}


def freeze(output, mode):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = {"frozen_at_utc": now(), "mode": mode, "files_sha256": scientific_files(),
              "scope": "Local prospective audit, not independent registration."}
    write_json(output / "FREEZE.json", frozen)
    write_json(output / "FROZEN_PROTOCOL.json", read_protocol())
    return output, frozen


def check_freeze(frozen):
    assert scientific_files() == frozen["files_sha256"], "Frozen input changed."


def checkpoint(model, optimizer, update, datasets, out, old_protocol):
    model.save(out / f"model_{update:04d}.npz")
    if optimizer is not None:
        moments = {f"m_{i}": v for i, v in enumerate(optimizer.m)}
        moments.update({f"v_{i}": v for i, v in enumerate(optimizer.v)})
        moments["step_count"] = np.array(optimizer.step_count)
        np.savez_compressed(out / f"optimizer_{update:04d}.npz", **moments)
    row = {"update": update, "recorded_at_utc": now()}
    for name in ("train", "calibration"):
        d = datasets[name]
        tr = base.extract(model, d["tokens"], batch_size=64)
        row[name] = base.measure(tr, d["labels"], d["family_ids"], old_protocol)
        # Full states are exactly regenerable from the saved weights/input arrays.
        np.savez_compressed(out / f"{name}_predictions_{update:04d}.npz", logits=tr["logits"], probabilities=tr["probabilities"])
        if name == "calibration":
            row["resume_errors"] = base.resume_audit(model, tr)
    print(f"{out.name} checkpoint {update:4d}: train={row['train']['accuracy_by_step'][-1]:.4f}, "
          f"calibration={row['calibration']['accuracy_by_step'][-1]:.4f}", flush=True)
    return row


def run_one(seed, arm, batches, data, panel_out, protocol, old_protocol):
    out = panel_out / f"seed_{seed}_{arm['name']}"
    out.mkdir(exist_ok=False)
    model = base.ArithmeticTransformer(old_protocol, seed=seed)
    initial = weight_digest(model)
    optimizer = MatrixAdamW(model.parameters(), weight_decay=arm["weight_decay"],
        learning_rate=protocol["preserved"]["learning_rate"], beta1=protocol["preserved"]["beta1"],
        beta2=protocol["preserved"]["beta2"], epsilon=protocol["preserved"]["epsilon"],
        max_grad_norm=protocol["preserved"]["gradient_clip"])
    metadata = {"seed": seed, "arm": arm["name"], "weight_decay": arm["weight_decay"],
        "initial_weight_sha256": initial, "batch_index_sha256": array_digest(batches),
        "parameter_count": sum(p.data.size for p in model.parameters()),
        "decayed_parameter_names": [p.name for p in optimizer.decayed],
        "started_at_utc": now()}
    write_json(out / "RUN_MANIFEST.json", metadata)
    started = time.perf_counter()
    checkpoints = [checkpoint(model, None, 0, data, out, old_protocol)]
    write_json(out / "checkpoints.json", checkpoints)
    metadata["training_started_at_utc"] = now()
    write_json(out / "RUN_MANIFEST.json", metadata)
    with (out / "training_log.jsonl").open("x") as log:
        for update, batch in enumerate(batches, start=1):
            _, logits, _ = model(data["train"]["tokens"][batch])
            loss = cross_entropy(logits[-1], data["train"]["labels"][batch])
            optimizer.zero_grad()
            loss.backward()
            if not np.isfinite(loss.data) or any(not np.all(np.isfinite(p.grad)) for p in optimizer.parameters):
                raise FloatingPointError(f"Non-finite objective/gradient: {seed}/{arm['name']}/{update}")
            norm = optimizer.step()
            if any(not np.all(np.isfinite(p.data)) for p in optimizer.parameters):
                raise FloatingPointError("Non-finite weights.")
            log.write(json.dumps({"update": update, "loss": float(loss.data), "grad_norm": norm,
                                  "elapsed_s": time.perf_counter()-started}) + "\n")
            if update in protocol["design"]["checkpoints"]:
                log.flush()
                checkpoints.append(checkpoint(model, optimizer, update, data, out, old_protocol))
                write_json(out / "checkpoints.json", checkpoints)
            elif update % protocol["design"]["logging_interval"] == 0:
                print(f"{out.name} update {update}/6000, batch NLL={float(loss.data):.4f}", flush=True)
            if update % 25 == 0:
                gc.collect()
    final = checkpoints[-1]["calibration"]
    eligible = [r["step"] for r in final["decision_change"] if r["eligible"]]
    result = {**metadata, "completed_at_utc": now(), "updates_completed": len(batches),
        "elapsed_seconds": time.perf_counter()-started, "checkpoints": checkpoints,
        "final_train_accuracy": checkpoints[-1]["train"]["accuracy_by_step"][-1],
        "final_calibration_accuracy": final["accuracy_by_step"][-1],
        "competence_passed": final["accuracy_by_step"][-1] >= protocol["measurements"]["competence_threshold"],
        "eligible_steps": eligible, "history_probes_fitted": 0}
    write_json(out / "result.json", result)
    return result


def decide(runs, protocol):
    regimes = []
    for arm in protocol["design"]["arms"]:
        these = [r for r in runs if r["arm"] == arm["name"]]
        complete = len(these) == 2 and all(r["updates_completed"] == 6000 for r in these)
        common = sorted(set.intersection(*(set(r["eligible_steps"]) for r in these))) if these else []
        competence = complete and all(r["competence_passed"] for r in these)
        target = len(common) >= protocol["measurements"]["min_common_eligible_steps"]
        regimes.append({"arm": arm["name"], "complete": complete, "competence_passed_all_seeds": competence,
                        "common_eligible_steps": common, "target_passed": target,
                        "ready_for_observer_design": bool(competence and target)})
    selected = next((r["arm"] for r in regimes if r["ready_for_observer_design"]), None)
    paired = []
    for seed in protocol["design"]["model_seeds"]:
        control = next(r for r in runs if r["seed"] == seed and r["arm"] == "no_decay")
        decay = next(r for r in runs if r["seed"] == seed and r["arm"] == "matrix_decay")
        assert control["initial_weight_sha256"] == decay["initial_weight_sha256"]
        assert control["batch_index_sha256"] == decay["batch_index_sha256"]
        paired.append({"seed": seed, "decay_minus_no_decay_accuracy":
                       decay["final_calibration_accuracy"] - control["final_calibration_accuracy"]})
    return {"regimens": regimes, "selected_development_candidate": selected, "paired_effects": paired,
            "mean_paired_accuracy_difference": float(np.mean([r["decay_minus_no_decay_accuracy"] for r in paired])),
            "status": "candidate_ready_for_observer_design" if selected else "no_regimen_ready",
            "history_probes_fitted": 0, "confirmation_opened": False, "sufficiency_hypothesis_tested": False}


def run_panel(mode, destination):
    protocol = read_protocol()
    old = validate_protocol(protocol)
    out, frozen = freeze(destination, mode)
    from checks_a0b import run_checks
    try:
        checks = run_checks()
        write_json(out / "method_checks.json", checks)
        if mode == "methods":
            check_freeze(frozen)
            print(json.dumps(checks, indent=2))
            return checks
        started = time.perf_counter()
        data, manifest = base.make_partitions(old)
        write_json(out / "partitions.json", manifest)
        assert manifest["example_counts"] == {"train": 2992, "calibration": 950}
        np.savez_compressed(out / "development_data.npz", **{
            f"{split}_{name}": values for split, d in data.items() for name, values in d.items()})
        streams = {}
        for seed in protocol["design"]["model_seeds"]:
            streams[seed] = np.random.default_rng(500000 + seed).integers(
                0, len(data["train"]["labels"]), size=(6000, 64), dtype=np.int64)
            np.savez_compressed(out / f"batch_stream_{seed}.npz", indices=streams[seed])
        runs = []
        for seed, name in protocol["design"]["run_order"]:
            check_freeze(frozen)
            arm = next(a for a in protocol["design"]["arms"] if a["name"] == name)
            runs.append(run_one(seed, arm, streams[seed], data, out, protocol, old))
            write_json(out / "completed_runs.json", runs)
        decision = decide(runs, protocol)
        check_freeze(frozen)
        result = {"study_id": protocol["study_id"], "version": protocol["version"],
                  "completed_at_utc": now(), "elapsed_seconds": time.perf_counter()-started,
                  "runs": runs, "decision": decision, "total_updates": sum(r["updates_completed"] for r in runs),
                  "reserved_examples_evaluated": 0, "old_models_loaded": False,
                  "frozen_sources_unchanged": True, "methods_passed": True,
                  "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()}}
        write_json(out / "result.json", result)
        print(json.dumps(decision, indent=2), flush=True)
        return result
    except Exception as exc:
        write_json(out / "INCOMPLETE.json", {"time_utc": now(), "error": repr(exc),
                                            "status": "incomplete_or_invalid_no_claim"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["methods", "panel"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_panel(args.mode, args.output)
