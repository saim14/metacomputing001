"""Synthetic optimizer, pairing and inherited implementation checks; no natural fits."""
import numpy as np

import run_a0b as run
from src.tiny_autograd import Adam, parameter
from test_methods import run_checks as old_checks


def run_checks():
    result = {"inherited_methods": old_checks()}
    p = run.read_protocol()
    old = run.validate_protocol(p)
    rng = np.random.default_rng(9011)
    # Zero-decay must be an EXACT extension of old Adam, including clipping/moments.
    initial = [rng.normal(size=(2, 3)).astype(np.float32), rng.normal(size=3).astype(np.float32)]
    a = [parameter(v.copy(), f"a{i}") for i, v in enumerate(initial)]
    b = [parameter(v.copy(), f"b{i}") for i, v in enumerate(initial)]
    adam = Adam(a, learning_rate=0.001)
    adamw0 = run.MatrixAdamW(b, learning_rate=0.001, weight_decay=0)
    for _ in range(10):
        for aa, bb in zip(a, b):
            gradient = rng.normal(size=aa.shape).astype(np.float32)
            aa.grad[...] = gradient
            bb.grad[...] = gradient
        assert adam.step() == adamw0.step()
        for aa, bb in zip(a, b):
            np.testing.assert_array_equal(aa.data, bb.data)
        for am, bm in zip(adam.m + adam.v, adamw0.m + adamw0.v):
            np.testing.assert_array_equal(am, bm)
    result["zero_decay_exact_original_adam_equivalence"] = True
    # Pure shrink on matrices with zero gradients; vector must be unchanged.
    matrix = parameter(np.array([[2.0, -3.0]], dtype=np.float32), "matrix")
    vector = parameter(np.array([2.0, -3.0], dtype=np.float32), "vector")
    opt = run.MatrixAdamW([matrix, vector], learning_rate=0.001, weight_decay=1)
    opt.step()
    np.testing.assert_allclose(matrix.data, np.array([[2.0, -3.0]]) * 0.999, atol=1e-7)
    np.testing.assert_array_equal(vector.data, [2.0, -3.0])
    result["zero_gradient_shrink_and_matrix_only_mask"] = True
    # Multistep hand calculation with pre-decay gradients, global clipping and bias correction.
    mats = [parameter(np.array([[1.5, -2.0]], dtype=np.float32), "w"),
            parameter(np.array([0.25, -0.5], dtype=np.float32), "b")]
    expected = [v.data.astype(np.float64).copy() for v in mats]
    m, v = [np.zeros_like(x) for x in expected], [np.zeros_like(x) for x in expected]
    opt = run.MatrixAdamW(mats, learning_rate=0.001, weight_decay=1)
    for t in range(1, 6):
        gradients = [rng.normal(size=x.shape) for x in expected]
        for x, g in zip(mats, gradients):
            x.grad[...] = g
        norm = np.sqrt(sum(np.sum(g*g) for g in gradients))
        for i, g in enumerate(gradients):
            gg = g * min(1, 1/(norm+1e-12))
            m[i] = 0.9*m[i] + 0.1*gg
            v[i] = 0.999*v[i] + 0.001*gg*gg
            if i == 0:
                expected[i] *= 0.999
            expected[i] -= 0.001*(m[i]/(1-0.9**t))/(np.sqrt(v[i]/(1-0.999**t))+1e-8)
        opt.step()
        for got, want in zip(mats, expected):
            np.testing.assert_allclose(got.data, want, atol=5e-7, rtol=2e-6)
    result["manual_multistep_decoupled_update"] = True
    hashes = []
    for seed in p["design"]["model_seeds"]:
        a, b = [run.base.ArithmeticTransformer(old, seed=seed) for _ in range(2)]
        assert run.weight_digest(a) == run.weight_digest(b)
        hashes.append(run.weight_digest(a))
        assert sum(x.data.size for x in a.parameters()) == 5873
        batch_a = np.random.default_rng(500000+seed).integers(0,2992,size=(6000,64),dtype=np.int64)
        batch_b = np.random.default_rng(500000+seed).integers(0,2992,size=(6000,64),dtype=np.int64)
        np.testing.assert_array_equal(batch_a,batch_b)
    assert len(set(hashes)) == 2
    result["paired_initial_weights_and_full_batch_streams"] = True
    data, partitions = run.base.make_partitions(old)
    assert set(data) == {"train", "calibration"}
    assert partitions["family_counts"] == {"train":581,"calibration":193,"reserve":195}
    result["reserve_unexpanded_and_old_data_partition_preserved"] = True
    # Decision fixture must reject one successful seed, and prefer simpler arm on ties.
    fake = []
    for seed in [2017,2027]:
        for arm in ["no_decay","matrix_decay"]:
            fake.append({"seed":seed,"arm":arm,"updates_completed":6000,
                         "eligible_steps":[2,3],"competence_passed":True,
                         "final_calibration_accuracy":0.95,
                         "initial_weight_sha256":str(seed),"batch_index_sha256":str(seed)})
    assert run.decide(fake,p)["selected_development_candidate"] == "no_decay"
    fake[0]["competence_passed"] = False
    assert run.decide(fake,p)["selected_development_candidate"] == "matrix_decay"
    fake[1]["competence_passed"] = False
    assert run.decide(fake,p)["selected_development_candidate"] is None
    result["all_seed_gate_and_fixed_selection_policy"] = True
    result["passed"] = True
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(run_checks(),indent=2))
