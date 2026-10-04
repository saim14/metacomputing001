"""Instrumented encoder-only Transformer for T1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tiny_autograd import Tensor, gelu, parameter, softmax


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    sequence_length: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    n_classes: int = 2
    layer_norm_epsilon: float = 1e-5


class Module:
    def parameters(self) -> list[Tensor]:
        result: list[Tensor] = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                result.append(value)
            elif isinstance(value, Module):
                result.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        result.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        result.append(item)
        return result


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator, name: str) -> None:
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = parameter(
            rng.normal(0.0, scale, size=(in_features, out_features)),
            f"{name}.weight",
        )
        self.bias = parameter(np.zeros(out_features), f"{name}.bias")

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias


class LayerNorm(Module):
    def __init__(self, features: int, epsilon: float, name: str) -> None:
        self.weight = parameter(np.ones(features), f"{name}.weight")
        self.bias = parameter(np.zeros(features), f"{name}.bias")
        self.epsilon = epsilon

    def __call__(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        variance = (centered * centered).mean(axis=-1, keepdims=True)
        normalized = centered / ((variance + self.epsilon) ** 0.5)
        return normalized * self.weight + self.bias


class TransformerBlock(Module):
    def __init__(self, config: ModelConfig, rng: np.random.Generator, index: int) -> None:
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        prefix = f"layers.{index}"
        self.config = config
        self.norm1 = LayerNorm(config.d_model, config.layer_norm_epsilon, f"{prefix}.norm1")
        self.q_proj = Linear(config.d_model, config.d_model, rng, f"{prefix}.q_proj")
        self.k_proj = Linear(config.d_model, config.d_model, rng, f"{prefix}.k_proj")
        self.v_proj = Linear(config.d_model, config.d_model, rng, f"{prefix}.v_proj")
        self.o_proj = Linear(config.d_model, config.d_model, rng, f"{prefix}.o_proj")
        self.norm2 = LayerNorm(config.d_model, config.layer_norm_epsilon, f"{prefix}.norm2")
        self.ff1 = Linear(config.d_model, config.d_ff, rng, f"{prefix}.ff1")
        self.ff2 = Linear(config.d_ff, config.d_model, rng, f"{prefix}.ff2")

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, width = x.shape
        heads = self.config.n_heads
        head_width = width // heads

        normalized = self.norm1(x)
        q = self.q_proj(normalized).reshape(batch, length, heads, head_width).transpose(0, 2, 1, 3)
        k = self.k_proj(normalized).reshape(batch, length, heads, head_width).transpose(0, 2, 1, 3)
        v = self.v_proj(normalized).reshape(batch, length, heads, head_width).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(float(head_width))
        attention = softmax(scores, axis=-1)
        attended = attention @ v
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
        x = x + self.o_proj(attended)
        x = x + self.ff2(gelu(self.ff1(self.norm2(x))))
        return x, attention


class TinyTransformer(Module):
    """A standard stack with a shared layer-wise readout for instrumentation."""

    def __init__(self, config: ModelConfig, seed: int) -> None:
        self.config = config
        rng = np.random.default_rng(seed)
        self.token_embedding = parameter(
            rng.normal(0.0, 0.02, size=(config.vocab_size, config.d_model)),
            "token_embedding",
        )
        self.position_embedding = parameter(
            rng.normal(0.0, 0.02, size=(config.sequence_length, config.d_model)),
            "position_embedding",
        )
        self.layers = [TransformerBlock(config, rng, i) for i in range(config.n_layers)]
        self.readout_norm = LayerNorm(config.d_model, config.layer_norm_epsilon, "readout_norm")
        self.readout = Linear(config.d_model, config.n_classes, rng, "readout")

    def _read(self, state: Tensor) -> Tensor:
        return self.readout(self.readout_norm(state[:, 0, :]))

    def __call__(
        self,
        tokens: np.ndarray,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        batch, length = tokens.shape
        if length != self.config.sequence_length:
            raise ValueError(f"expected sequence length {self.config.sequence_length}, got {length}")
        one_hot = np.eye(self.config.vocab_size, dtype=np.float32)[tokens]
        state = Tensor(one_hot) @ self.token_embedding + self.position_embedding
        states = [state]
        logits = [self._read(state)]
        attentions: list[Tensor] = []
        for layer in self.layers:
            state, attention = layer(state)
            states.append(state)
            logits.append(self._read(state))
            attentions.append(attention)
        return states, logits, attentions

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            parameter_tensor.name or f"parameter_{index}": parameter_tensor.data.copy()
            for index, parameter_tensor in enumerate(self.parameters())
        }

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **self.state_dict())


def numpy_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)
