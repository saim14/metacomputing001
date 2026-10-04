"""P1-A0: bounded arithmetic readiness pilot; no history-probe fitting.

Run: python experiment.py methods --output /new/output/path
     python experiment.py pilot --output /different/new/output/path
Only NumPy and the Python standard library are required.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import platform
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "t1_original"))
from src.tiny_autograd import Adam, Tensor, cross_entropy
from src.tiny_transformer import ModelConfig, TinyTransformer, numpy_softmax


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_protocol():
    return json.loads((ROOT / "P1_A0_protocol.json").read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArithmeticTransformer(TinyTransformer):
    """Original Transformer block, reused at every computation step."""
    def __init__(self, protocol=None, *, seed=None):
        protocol = read_protocol() if protocol is None else protocol
        spec, data = protocol["model"], protocol["data"]
        config = ModelConfig(
            vocab_size=data["modulus"] + 1,
            sequence_length=data["operands"] + 1,
            d_model=spec["d_model"], n_heads=spec["n_heads"],
            d_ff=spec["d_ff"], n_layers=1, n_classes=data["modulus"],
            layer_norm_epsilon=spec["layer_norm_epsilon"],
        )
        model_seed = spec["seed"] if seed is None else seed
        if model_seed in protocol["scope"]["old_reserved_seeds_forbidden"]:
            raise ValueError("The old reserved seeds remain sealed.")
        super().__init__(config, model_seed)
        self.refinements = spec["n_steps"]

    def __call__(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] != self.config.sequence_length:
            raise ValueError("Expected [batch, CLS + operands].")
        if not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError("Token IDs must be integers.")
        if np.any(tokens < 0) or np.any(tokens >= self.config.vocab_size):
            raise ValueError("Token ID outside vocabulary.")
        one_hot = np.eye(self.config.vocab_size, dtype=np.float32)[tokens]
        state = Tensor(one_hot) @ self.token_embedding + self.position_embedding
        states, logits, attentions = [state], [self._read(state)], []
        for _ in range(self.refinements):
            state, attention = self.layers[0](state)
            states.append(state)
            logits.append(self._read(state))
            attentions.append(attention)
        return states, logits, attentions

    def resume(self, state, step):
        """Continue from all tokens; weights and the step index stay fixed."""
        if not 0 <= step <= self.refinements:
            raise ValueError("Invalid computation step.")
        expected = (self.config.sequence_length, self.config.d_model)
        if np.asarray(state).ndim != 3 or tuple(state.shape[1:]) != expected:
            raise ValueError("Resume requires complete all-token state.")
        current = Tensor(np.asarray(state, dtype=np.float32).copy())
        for _ in range(step, self.refinements):
            current, _ = self.layers[0](current)
        return self._read(current).data.copy()

    def load(self, path):
        with np.load(path, allow_pickle=False) as saved:
            params = {p.name: p for p in self.parameters()}
            if set(saved.files) != set(params):
                raise ValueError("Checkpoint key mismatch.")
            for name, parameter in params.items():
                if parameter.shape != saved[name].shape:
                    raise ValueError("Checkpoint shape mismatch.")
                parameter.data[...] = saved[name]


@contextmanager
def inference(model):
    """Avoid autograd graphs without modifying source or parameter values."""
    parameters = model.parameters()
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad = False
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad = flag


def make_partitions(protocol=None):
    """Expand train/calibration only; reserve rows are never instantiated."""
    protocol = read_protocol() if protocol is None else protocol
    spec = protocol["data"]
    families = list(itertools.combinations_with_replacement(
        range(spec["modulus"]), spec["operands"]))
    order = np.random.default_rng(spec["partition_seed"]).permutation(len(families))
    n_train = int(spec["train_fraction"] * len(families))
    n_cal = int(spec["calibration_fraction"] * len(families))
    ids = {"train": order[:n_train], "calibration": order[n_train:n_train+n_cal],
           "reserve": order[n_train+n_cal:]}
    datasets = {}
    for name in ("train", "calibration"):
        tokens, labels, family_ids = [], [], []
        for family_id in sorted(ids[name].tolist()):
            for operands in sorted(set(itertools.permutations(families[family_id]))):
                tokens.append([spec["modulus"], *operands])
                labels.append(sum(operands) % spec["modulus"])
                family_ids.append(family_id)
        datasets[name] = {
            "tokens": np.asarray(tokens, dtype=np.int64),
            "labels": np.asarray(labels, dtype=np.int64),
            "family_ids": np.asarray(family_ids, dtype=np.int64),
        }
    manifest = {"family_ids": {k: v.tolist() for k, v in ids.items()},
                "family_counts": {k: len(v) for k, v in ids.items()},
                "example_counts": {k: len(v["labels"]) for k, v in datasets.items()},
                "reserve_examples_constructed": False,
                "total_canonical_families": len(families)}
    return datasets, manifest


def extract(model, tokens, batch_size=64):
    states, logits = [], []
    with inference(model):
        for start in range(0, len(tokens), batch_size):
            ss, ll, _ = model(tokens[start:start+batch_size])
            states.append(np.stack([s.data for s in ss], axis=1))
            logits.append(np.stack([v.data for v in ll], axis=1))
    state_array = np.concatenate(states)
    logit_array = np.concatenate(logits)
    return {"states": state_array, "logits": logit_array,
            "probabilities": numpy_softmax(logit_array)}


def observable_features(trajectory, step, view="all_token", history=False):
    """Extraction interface only: never accesses labels or future steps."""
    if view not in {"cls", "all_token"}:
        raise ValueError("Unknown observation view.")
    if not 1 <= step < trajectory["states"].shape[1] - 1:
        raise ValueError("Use an intermediate step.")

    def at(index, chosen_view):
        # Slice BEFORE calculations: changing future arrays cannot change features.
        state = trajectory["states"][:, index]
        probability = trajectory["probabilities"][:, index]
        entropy = -(probability * np.log(np.clip(probability, 1e-8, 1))).sum(1, keepdims=True)
        ordered = np.sort(probability, axis=1)
        margin = (ordered[:, -1] - ordered[:, -2])[:, None]
        values = state[:, 0] if chosen_view == "cls" else state.reshape(len(state), -1)
        return np.concatenate([values, probability, entropy, margin], axis=1)

    current = at(step, view)
    if not history or step == 1:
        return current
    past = [at(index, "cls") for index in range(1, step)]
    return np.concatenate([current, *past], axis=1)


def resume_audit(model, trajectory, count=16):
    result = {}
    expected = trajectory["logits"][:count, -1]
    with inference(model):
        for step in range(model.refinements + 1):
            resumed = model.resume(trajectory["states"][:count, step], step)
            error = float(np.max(np.abs(resumed - expected)))
            if not np.allclose(resumed, expected, atol=2e-5, rtol=2e-5):
                raise AssertionError(f"State resumption failed at step {step}: {error}")
            result[str(step)] = error
    return result


def measure(trajectory, labels, family_ids, protocol):
    predictions = trajectory["logits"].argmax(axis=-1)
    accuracy = (predictions == labels[:, None]).mean(axis=0)
    rows = []
    spec = protocol["measurements"]
    for step in spec["eligible_candidate_steps"]:
        flips = predictions[:, step] != predictions[:, -1]
        counts = [int(np.sum(flips == value)) for value in (False, True)]
        families = [len(np.unique(family_ids[flips == value])) for value in (False, True)]
        eligible = (min(counts) >= spec["target_gate_min_events_per_class"] and
                    min(families) >= spec["target_gate_min_families_per_class"] and
                    min(counts) / len(flips) >= spec["target_gate_min_class_fraction"])
        rows.append({"step": step, "unchanged_examples": counts[0], "changed_examples": counts[1],
                     "unchanged_families": families[0], "changed_families": families[1],
                     "flip_prevalence": float(flips.mean()), "eligible": bool(eligible)})
    return {"accuracy_by_step": accuracy.tolist(), "decision_change": rows}


def freeze(output, mode):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)  # refuse to overwrite a prior run
    files = ["P1_A0_protocol.json", "P1_A0_Protocol.md", "experiment.py", "test_methods.py",
             "vendor/T1_source_0.1.0.zip", "vendor/t1_original/src/tiny_autograd.py",
             "vendor/t1_original/src/tiny_transformer.py"]
    record = {"frozen_at_utc": utc_now(), "mode": mode,
              "files_sha256": {name: sha256(ROOT / name) for name in files},
              "scope": "Local prospective freeze, not independent registration."}
    write_json(output / "FREEZE.json", record)
    shutil.copyfile(ROOT / "P1_A0_protocol.json", output / "FROZEN_PROTOCOL.json")
    return output, record


def verify_freeze(record):
    for relative, expected in record["files_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise AssertionError(f"Frozen scientific input changed: {relative}")


def checkpoint(model, optimizer, update, datasets, output, protocol):
    model.save(output / f"model_update_{update:04d}.npz")
    if optimizer is not None:
        moments = {f"m_{i}": m for i, m in enumerate(optimizer.m)}
        moments.update({f"v_{i}": v for i, v in enumerate(optimizer.v)})
        moments["step_count"] = np.array(optimizer.step_count)
        np.savez_compressed(output / f"optimizer_update_{update:04d}.npz", **moments)
    row = {"update": update, "recorded_at_utc": utc_now()}
    for name in ("train", "calibration"):
        data = datasets[name]
        trajectory = extract(model, data["tokens"], protocol["training"]["evaluation_batch_size"])
        row[name] = measure(trajectory, data["labels"], data["family_ids"], protocol)
        if name == "calibration":
            row["resume_errors"] = resume_audit(model, trajectory)
            np.savez_compressed(output / f"calibration_trajectory_{update:04d}.npz", **trajectory)
    print(f"checkpoint {update:4d}: train={row['train']['accuracy_by_step'][-1]:.4f} "
          f"calibration={row['calibration']['accuracy_by_step'][-1]:.4f}", flush=True)
    return row


def run(mode, output):
    protocol = read_protocol()
    output, frozen = freeze(output, mode)
    from test_methods import run_checks
    try:
        checks = run_checks()
        write_json(output / "method_checks.json", checks)
        if mode == "methods":
            verify_freeze(frozen)
            print(json.dumps(checks, indent=2), flush=True)
            return checks
        if mode != "pilot":
            raise ValueError("Unknown mode.")
        started = time.perf_counter()
        datasets, partitions = make_partitions(protocol)
        write_json(output / "partitions.json", partitions)
        np.savez_compressed(output / "development_data.npz", **{
            f"{name}_{key}": value for name, data in datasets.items() for key, value in data.items()
        })
        model = ArithmeticTransformer(protocol)
        spec = protocol["training"]
        optimizer = Adam(model.parameters(), learning_rate=spec["learning_rate"],
                         beta1=spec["beta1"], beta2=spec["beta2"], epsilon=spec["epsilon"],
                         max_grad_norm=spec["max_grad_norm"])
        rng = np.random.default_rng(spec["sampling_seed"])
        checkpoints = [checkpoint(model, None, 0, datasets, output, protocol)]
        write_json(output / "checkpoints.json", checkpoints)
        train = datasets["train"]
        training_started = utc_now()
        with (output / "training_log.jsonl").open("x") as log:
            for update in range(1, spec["updates"] + 1):
                batch = rng.integers(0, len(train["labels"]), size=spec["batch_size"])
                _, logits, _ = model(train["tokens"][batch])
                loss = cross_entropy(logits[-1], train["labels"][batch])
                optimizer.zero_grad()
                loss.backward()
                if not np.isfinite(loss.data) or any(not np.all(np.isfinite(p.grad)) for p in optimizer.parameters):
                    raise FloatingPointError(f"Non-finite objective/gradient at update {update}")
                norm = optimizer.step()
                if any(not np.all(np.isfinite(p.data)) for p in optimizer.parameters):
                    raise FloatingPointError(f"Non-finite parameters at update {update}")
                log.write(json.dumps({"update": update, "loss": float(loss.data),
                                      "grad_norm": norm, "elapsed_s": time.perf_counter()-started}) + "\n")
                if update in spec["checkpoints"]:
                    log.flush()
                    checkpoints.append(checkpoint(model, optimizer, update, datasets, output, protocol))
                    write_json(output / "checkpoints.json", checkpoints)
                elif update % spec["progress_every"] == 0:
                    print(f"update {update}/{spec['updates']}, batch NLL={float(loss.data):.4f}", flush=True)
                if update % 25 == 0:
                    gc.collect()
        final = checkpoints[-1]["calibration"]
        competence = final["accuracy_by_step"][-1] >= protocol["measurements"]["competence_gate"]
        eligible = [r["step"] for r in final["decision_change"] if r["eligible"]]
        target = len(eligible) >= protocol["measurements"]["target_gate_min_eligible_steps"]
        status = ("competence_failed" if not competence else
                  "target_failed" if not target else "ready_for_observer_design")
        verify_freeze(frozen)
        result = {
            "study_id": protocol["study_id"], "version": protocol["version"],
            "status": status, "methods_passed": True,
            "seed": protocol["model"]["seed"], "updates_completed": spec["updates"],
            "training_started_at_utc": training_started, "completed_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter()-started,
            "parameter_count": sum(p.data.size for p in model.parameters()),
            "competence_passed": bool(competence), "target_passed": bool(target),
            "eligible_steps": eligible, "final_calibration_accuracy": final["accuracy_by_step"][-1],
            "checkpoints": checkpoints, "partitions": partitions,
            "history_probes_fitted": 0, "reserved_examples_evaluated": 0,
            "old_models_loaded": False, "confirmation_opened": False,
            "sufficiency_hypothesis_tested": False,
            "frozen_inputs_unchanged": True,
            "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
            "model_config": asdict(model.config), "refinement_steps": model.refinements,
        }
        write_json(output / "result.json", result)
        print(f"P1-A0 complete: {status}; no history observers were fitted.", flush=True)
        return result
    except Exception as exc:
        write_json(output / "INCOMPLETE.json", {"time_utc": utc_now(), "error": repr(exc),
                   "status": "invalid_or_incomplete_no_scientific_interpretation"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["methods", "pilot"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.output)
