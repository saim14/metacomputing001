"""Executable method audits; no natural model optimization or probe fits."""
import itertools
from pathlib import Path
import tempfile

import numpy as np

import experiment as ex
from src.tiny_autograd import Tensor, cross_entropy, parameter, Adam


def run_checks():
    checks = {}
    protocol = ex.read_protocol()
    data, manifest = ex.make_partitions(protocol)
    ids = {name: set(values) for name, values in manifest["family_ids"].items()}
    assert len(set.union(*ids.values())) == 969
    assert all(not ids[a] & ids[b] for a, b in itertools.combinations(ids, 2))
    assert set(data) == {"train", "calibration"} and not manifest["reserve_examples_constructed"]
    for split, values in data.items():
        tokens, labels, fids = values["tokens"], values["labels"], values["family_ids"]
        assert np.array_equal(labels, tokens[:, 1:].sum(axis=1) % 17)
        assert np.all(tokens[:, 0] == 17)
        assert len(np.unique(tokens, axis=0)) == len(tokens)
        assert set(fids.tolist()) == ids[split]
        for family in np.unique(fids):
            rows = tokens[fids == family, 1:]
            assert np.all(np.sort(rows, axis=1) == np.sort(rows[0]))
    again, again_manifest = ex.make_partitions(protocol)
    assert again_manifest == manifest
    assert all(np.array_equal(data[s][k], again[s][k]) for s in data for k in data[s])
    checks["exact_labels_family_separation_and_repeatability"] = True
    checks["reserved_examples_not_constructed"] = True

    model = ex.ArithmeticTransformer(protocol, seed=8001)
    params = model.parameters()
    assert len(params) == len(set(map(id, params)))
    assert len(params) == len({p.name for p in params}) and len(model.layers) == 1
    tokens = data["train"]["tokens"][:4]
    labels = data["train"]["labels"][:4]
    trajectory = ex.extract(model, tokens)
    assert trajectory["states"].shape == (4, 7, 4, 24)
    assert trajectory["probabilities"].shape == (4, 7, 17)
    np.testing.assert_allclose(trajectory["probabilities"].sum(-1), 1, atol=2e-6)
    checks["one_shared_block_unique_parameters_and_shapes"] = True
    checks["normalization"] = True
    checks["state_resume_max_errors"] = ex.resume_audit(model, trajectory, count=4)
    try:
        with ex.inference(model):
            model.resume(trajectory["states"][:, 2, 0], 2)
        raise AssertionError("CLS-only resume should be rejected.")
    except ValueError:
        pass
    checks["partial_state_resume_rejected"] = True

    for view, width in (("cls", 43), ("all_token", 115)):
        for step in (2, 3, 4, 5):
            current = ex.observable_features(trajectory, step, view, history=False)
            history = ex.observable_features(trajectory, step, view, history=True)
            assert current.shape[1] == width
            assert history.shape[1] == width + (step - 1) * 43
            np.testing.assert_array_equal(current, history[:, :width])
            changed = {key: value.copy() for key, value in trajectory.items()}
            changed["states"][:, step+1:] = 12345
            changed["probabilities"][:, step+1:] = -12345
            changed["logits"][:] = 0  # logits are not used by observer extraction
            changed["labels"] = np.arange(4)  # labels are not used either
            np.testing.assert_array_equal(history, ex.observable_features(changed, step, view, history=True))
    checks["current_prefix_identical_no_future_or_label_leakage"] = True

    # Finite differences audit gradients through the block reused SIX times.
    _, logits, _ = model(tokens)
    loss = cross_entropy(logits[-1], labels)
    for p in params:
        p.zero_grad()
    loss.backward()
    assert all(np.all(np.isfinite(p.grad)) for p in params)
    errors = []
    for name in ("layers.0.q_proj.weight", "layers.0.ff1.weight", "readout.weight"):
        p = next(p for p in params if p.name == name)
        index = np.unravel_index(np.argmax(np.abs(p.grad)), p.shape)
        analytic = float(p.grad[index])
        original = float(p.data[index])
        epsilon = 1e-3
        values = []
        for sign in (1, -1):
            p.data[index] = original + sign * epsilon
            with ex.inference(model):
                _, candidate_logits, _ = model(tokens)
                values.append(float(cross_entropy(candidate_logits[-1], labels).data))
        p.data[index] = original
        numeric = (values[0] - values[1]) / (2 * epsilon)
        assert np.isclose(analytic, numeric, rtol=0.05, atol=0.006), (name, analytic, numeric)
        errors.append({"parameter": name, "index": [int(i) for i in index],
                       "analytic": analytic, "finite_difference": numeric})
    checks["tied_gradient_finite_differences"] = errors

    with tempfile.TemporaryDirectory(prefix="p1_method_") as temp:
        checkpoint = Path(temp) / "model.npz"
        model.save(checkpoint)
        loaded = ex.ArithmeticTransformer(protocol, seed=8002)
        loaded.load(checkpoint)
        a, b = ex.extract(model, tokens), ex.extract(loaded, tokens)
        np.testing.assert_array_equal(a["logits"], b["logits"])
    checks["serialization_exact_roundtrip"] = True

    # Only scalar synthetic optimizer fixture; not natural model training.
    scalar = parameter(np.array([2.0]), "fixture")
    optimizer = Adam([scalar], learning_rate=0.1)
    before = float(scalar.data[0] ** 2)
    for _ in range(10):
        optimizer.zero_grad()
        objective = (scalar * scalar).sum()
        objective.backward()
        optimizer.step()
    assert float(scalar.data[0] ** 2) < before
    checks["synthetic_scalar_optimizer"] = True

    # Exact oracle fixtures: current observation u vs full state (u,v).
    truth = np.array(list(itertools.product([0, 1], repeat=2)))
    y = truth[:, 0] ^ truth[:, 1]
    partial = np.array([y[truth[:, 0] == u].mean() for u in truth[:, 0]])
    full = y.astype(float)
    partial_brier = float(np.mean((partial - y) ** 2))
    full_brier = float(np.mean((full - y) ** 2))
    assert partial_brier == 0.25 and full_brier == 0.0
    checks["exact_oracle_controls"] = {
        "partial_current_brier": partial_brier,
        "partial_current_plus_missing_history_brier": full_brier,
        "complete_current_brier": full_brier,
        "complete_current_plus_redundant_history_brier": full_brier,
        "scope": "Truth-table arithmetic, not fitted-probe power or Transformer evidence."}
    checks["passed"] = True
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(run_checks(), indent=2))
