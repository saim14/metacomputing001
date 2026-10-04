from __future__ import annotations

import numpy as np
from pathlib import Path

from src.t1_experiment import (
    build_attention_control_features,
    generate_chain_lookup_task,
    generate_two_hop_counterfactual_suite,
)
from src.tiny_transformer import ModelConfig, TinyTransformer

from run_t1_u7b import load_config as load_u7b_config, validate_protocol as validate_u7b_protocol


def test_chain_task_balances_value_token_counts() -> None:
    key_count = 12
    tokens, labels = generate_chain_lookup_task(
        200,
        key_count=key_count,
        pair_count=6,
        chain_hops=3,
        rng=np.random.default_rng(19),
    )
    zero_counts = np.sum(tokens == key_count, axis=1)
    one_counts = np.sum(tokens == key_count + 1, axis=1)
    assert np.all(zero_counts == 2)
    assert np.all(one_counts == 2)
    assert 0.4 < np.mean(labels) < 0.6


def test_attention_controls_match_ordered_dimension() -> None:
    rng = np.random.default_rng(5)
    state = rng.normal(size=(30, 7, 36)).astype(np.float32)
    attention = rng.normal(size=(30, 6, 4, 3)).astype(np.float32)
    controls = build_attention_control_features(state, attention, layer=4, rng=rng)
    ordered_dimension = controls["ordered_attention_history"].shape[1]
    for name in (
        "attention_dimension_matched",
        "shuffled_attention_history",
        "sample_mismatched_attention_history",
    ):
        assert controls[name].shape[1] == ordered_dimension
    assert controls["current_plus_current_attention"].shape[1] < ordered_dimension


def test_structural_embeddings_enter_transformer_state() -> None:
    config = ModelConfig(
        vocab_size=15,
        sequence_length=14,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=2,
        use_structure_embeddings=True,
    )
    model = TinyTransformer(config, seed=13)
    tokens, _ = generate_chain_lookup_task(
        4,
        key_count=12,
        pair_count=6,
        chain_hops=3,
        rng=np.random.default_rng(29),
    )
    states, logits, attentions = model(tokens)
    names = {parameter.name for parameter in model.parameters()}
    assert "role_embedding" in names
    assert "pair_embedding" in names
    assert len(states) == 3 and len(logits) == 3 and len(attentions) == 2


def test_tied_transformer_reuses_one_block() -> None:
    config = ModelConfig(
        vocab_size=11,
        sequence_length=10,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=6,
        use_structure_embeddings=True,
        tie_layers=True,
    )
    model = TinyTransformer(config, seed=31)
    assert len(model.layers) == 6
    assert len({id(layer) for layer in model.layers}) == 1
    unique_parameter_ids = {id(parameter) for parameter in model.parameters()}

    untied_config = ModelConfig(
        vocab_size=11,
        sequence_length=10,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=6,
        use_structure_embeddings=True,
        tie_layers=False,
    )
    untied_model = TinyTransformer(untied_config, seed=31)
    untied_parameter_ids = {id(parameter) for parameter in untied_model.parameters()}
    assert len(unique_parameter_ids) < len(untied_parameter_ids)


def test_step_embeddings_parameterize_recurrent_time() -> None:
    base = dict(
        vocab_size=11,
        sequence_length=10,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=4,
        use_structure_embeddings=True,
        tie_layers=True,
    )
    timed_model = TinyTransformer(ModelConfig(**base, use_step_embeddings=True), seed=43)
    untimed_model = TinyTransformer(ModelConfig(**base, use_step_embeddings=False), seed=43)
    names = {parameter.name for parameter in timed_model.parameters()}
    assert "step_embedding" in names
    assert timed_model.step_embedding.shape == (4, 16)

    tokens, _ = generate_chain_lookup_task(
        4,
        key_count=8,
        pair_count=4,
        chain_hops=1,
        rng=np.random.default_rng(47),
    )
    timed_states, _, _ = timed_model(tokens)
    untimed_states, _, _ = untimed_model(tokens)
    assert not np.allclose(timed_states[-1].data, untimed_states[-1].data)


def test_residual_scale_changes_dynamics_without_changing_parameters() -> None:
    base = dict(
        vocab_size=11,
        sequence_length=10,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=4,
        use_structure_embeddings=True,
        tie_layers=True,
    )
    full_model = TinyTransformer(ModelConfig(**base, residual_scale=1.0), seed=53)
    half_model = TinyTransformer(ModelConfig(**base, residual_scale=0.5), seed=53)
    tokens, _ = generate_chain_lookup_task(
        4,
        key_count=8,
        pair_count=4,
        chain_hops=1,
        rng=np.random.default_rng(59),
    )
    full_states, _, _ = full_model(tokens)
    half_states, _, _ = half_model(tokens)

    assert set(full_model.state_dict()) == set(half_model.state_dict())
    for name in full_model.state_dict():
        assert np.array_equal(full_model.state_dict()[name], half_model.state_dict()[name])
    assert not np.allclose(full_states[-1].data, half_states[-1].data)


def test_default_residual_scale_preserves_prior_forward_pass() -> None:
    base = dict(
        vocab_size=11,
        sequence_length=10,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=3,
        use_structure_embeddings=True,
        tie_layers=True,
    )
    default_model = TinyTransformer(ModelConfig(**base), seed=61)
    explicit_model = TinyTransformer(ModelConfig(**base, residual_scale=1.0), seed=61)
    tokens, _ = generate_chain_lookup_task(
        4,
        key_count=8,
        pair_count=4,
        chain_hops=1,
        rng=np.random.default_rng(67),
    )
    default_states, default_logits, _ = default_model(tokens)
    explicit_states, explicit_logits, _ = explicit_model(tokens)
    assert np.array_equal(default_states[-1].data, explicit_states[-1].data)
    assert np.array_equal(default_logits[-1].data, explicit_logits[-1].data)


def test_constant_residual_schedule_matches_scalar_scale() -> None:
    base = dict(
        vocab_size=11,
        sequence_length=10,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=4,
        use_structure_embeddings=True,
        tie_layers=True,
        residual_scale=0.5,
    )
    scalar_model = TinyTransformer(ModelConfig(**base), seed=71)
    scheduled_model = TinyTransformer(
        ModelConfig(**base, residual_scale_schedule=(0.5, 0.5, 0.5, 0.5)),
        seed=71,
    )
    tokens, _ = generate_chain_lookup_task(
        4,
        key_count=8,
        pair_count=4,
        chain_hops=1,
        rng=np.random.default_rng(73),
    )
    scalar_states, scalar_logits, _ = scalar_model(tokens)
    scheduled_states, scheduled_logits, _ = scheduled_model(tokens)
    for scalar, scheduled in zip(scalar_states, scheduled_states):
        assert np.array_equal(scalar.data, scheduled.data)
    for scalar, scheduled in zip(scalar_logits, scheduled_logits):
        assert np.array_equal(scalar.data, scheduled.data)


def test_residual_schedule_order_changes_dynamics_not_parameters() -> None:
    base = dict(
        vocab_size=11,
        sequence_length=10,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=4,
        use_structure_embeddings=True,
        tie_layers=True,
        residual_scale=0.5,
    )
    ascending = TinyTransformer(
        ModelConfig(**base, residual_scale_schedule=(0.2, 0.4, 0.6, 0.8)),
        seed=79,
    )
    descending = TinyTransformer(
        ModelConfig(**base, residual_scale_schedule=(0.8, 0.6, 0.4, 0.2)),
        seed=79,
    )
    tokens, _ = generate_chain_lookup_task(
        4,
        key_count=8,
        pair_count=4,
        chain_hops=1,
        rng=np.random.default_rng(83),
    )
    ascending_states, _, _ = ascending(tokens)
    descending_states, _, _ = descending(tokens)
    assert set(ascending.state_dict()) == set(descending.state_dict())
    for name in ascending.state_dict():
        assert np.array_equal(ascending.state_dict()[name], descending.state_dict()[name])
    assert not np.allclose(ascending_states[-1].data, descending_states[-1].data)


def _follow_two_hop(tokens: np.ndarray, key_count: int) -> np.ndarray:
    answers = []
    for row in tokens:
        mapping = dict(zip(row[2::2].tolist(), row[3::2].tolist()))
        query = int(row[1])
        answers.append(int(mapping[mapping[query]]) - key_count)
    return np.asarray(answers, dtype=np.int64)


def test_two_hop_counterfactual_suite_is_balanced_and_matched() -> None:
    key_count = 8
    suite = generate_two_hop_counterfactual_suite(
        400,
        key_count=key_count,
        pair_count=5,
        rng=np.random.default_rng(89),
    )
    clean = suite["clean_tokens"]
    counterfactual = suite["counterfactual_tokens"]
    labels = suite["clean_labels"]
    counterfactual_labels = suite["counterfactual_labels"]

    assert np.array_equal(_follow_two_hop(clean, key_count), labels)
    assert np.array_equal(
        _follow_two_hop(counterfactual, key_count), counterfactual_labels
    )
    assert np.array_equal(counterfactual_labels, 1 - labels)
    assert np.array_equal(clean[:, :2], counterfactual[:, :2])
    assert np.array_equal(clean[:, 2::2], counterfactual[:, 2::2])
    for values in (clean, counterfactual):
        assert np.all(np.sum(values == key_count, axis=1) == 2)
        assert np.all(np.sum(values == key_count + 1, axis=1) == 2)


def test_two_hop_edge_corruptions_remove_the_required_path() -> None:
    key_count = 8
    suite = generate_two_hop_counterfactual_suite(
        2000,
        key_count=key_count,
        pair_count=5,
        rng=np.random.default_rng(97),
    )
    clean = suite["clean_tokens"]
    first = suite["first_edge_corrupted_tokens"]
    second = suite["second_edge_corrupted_tokens"]
    labels = suite["clean_labels"]

    for row in first:
        mapping = dict(zip(row[2::2].tolist(), row[3::2].tolist()))
        assert int(mapping[int(row[1])]) not in mapping

    random_answers = suite["second_edge_random_answers"]
    assert 0.46 <= float(np.mean(random_answers == labels)) <= 0.54
    assert np.array_equal(
        np.sort(clean[:, 2:], axis=1), np.sort(second[:, 2:], axis=1)
    )


def test_u7b_only_appends_preregistered_fresh_streams() -> None:
    study = load_u7b_config(Path("configs/t1_u7b_fresh_stream_extension.yaml"))
    reference = load_u7b_config(Path("configs/t1_u7_two_hop_gate.yaml"))
    verification = validate_u7b_protocol(study, reference)
    assert verification["base_updates"] == 4000
    assert verification["extension_updates"] == 2000
    assert verification["extension_stream_ids"] == [2, 3, 4, 5]
    assert verification["total_updates"] == 6000
