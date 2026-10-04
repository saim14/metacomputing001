from __future__ import annotations

import numpy as np

from src.t1_experiment import (
    build_attention_control_features,
    generate_chain_lookup_task,
)
from src.tiny_transformer import ModelConfig, TinyTransformer


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
