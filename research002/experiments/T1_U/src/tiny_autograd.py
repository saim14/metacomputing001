"""A small NumPy reverse-mode autodiff engine for the T1 experiment.

It intentionally implements only the tensor operations used by the tiny
Transformer.  The experiment therefore has no deep-learning-framework
dependency and remains portable to a constrained CPU environment.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _as_float_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum a broadcast gradient back to an operand's original shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A differentiable NumPy array."""

    def __init__(
        self,
        data: object,
        *,
        requires_grad: bool = False,
        name: str | None = None,
        _children: Sequence["Tensor"] = (),
    ) -> None:
        self.data = _as_float_array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.name = name
        self._prev = tuple(child for child in _children if child.requires_grad)
        self._backward = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @staticmethod
    def ensure(value: object) -> "Tensor":
        return value if isinstance(value, Tensor) else Tensor(value)

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad.fill(0.0)

    def backward(self) -> None:
        if self.data.size != 1:
            raise ValueError("backward() requires a scalar Tensor")
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other: object) -> "Tensor":
        other = Tensor.ensure(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
        )

        def backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = backward
        return out

    def __radd__(self, other: object) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: object) -> "Tensor":
        return self + (-Tensor.ensure(other))

    def __rsub__(self, other: object) -> "Tensor":
        return Tensor.ensure(other) - self

    def __mul__(self, other: object) -> "Tensor":
        other = Tensor.ensure(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
        )

        def backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.shape)

        out._backward = backward
        return out

    def __rmul__(self, other: object) -> "Tensor":
        return self * other

    def __truediv__(self, other: object) -> "Tensor":
        return self * (Tensor.ensure(other) ** -1.0)

    def __rtruediv__(self, other: object) -> "Tensor":
        return Tensor.ensure(other) / self

    def __pow__(self, exponent: float) -> "Tensor":
        out = Tensor(
            self.data**exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * exponent * (self.data ** (exponent - 1.0))

        out._backward = backward
        return out

    def __matmul__(self, other: object) -> "Tensor":
        other = Tensor.ensure(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
        )

        def backward() -> None:
            if self.requires_grad:
                grad = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += _unbroadcast(grad, self.shape)
            if other.requires_grad:
                grad = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += _unbroadcast(grad, other.shape)

        out._backward = backward
        return out

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def backward() -> None:
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                for current_axis in sorted(axes):
                    grad = np.expand_dims(grad, current_axis)
            self.grad += np.broadcast_to(grad, self.shape)

        out._backward = backward
        return out

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        if axis is None:
            divisor = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            divisor = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / float(divisor)

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)

        out._backward = backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        out = Tensor(
            self.data.transpose(*axes),
            requires_grad=self.requires_grad,
            _children=(self,),
        )
        inverse = np.argsort(axes)

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.transpose(*inverse)

        out._backward = backward
        return out

    def __getitem__(self, index: object) -> "Tensor":
        out = Tensor(
            self.data[index],
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def backward() -> None:
            if self.requires_grad:
                np.add.at(self.grad, index, out.grad)

        out._backward = backward
        return out

    def exp(self) -> "Tensor":
        values = np.exp(self.data)
        out = Tensor(values, requires_grad=self.requires_grad, _children=(self,))

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * values

        out._backward = backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = backward
        return out

    def tanh(self) -> "Tensor":
        values = np.tanh(self.data)
        out = Tensor(values, requires_grad=self.requires_grad, _children=(self,))

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (1.0 - values**2)

        out._backward = backward
        return out


def parameter(data: object, name: str) -> Tensor:
    return Tensor(data, requires_grad=True, name=name)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    stable = x - np.max(x.data, axis=axis, keepdims=True)
    exp_values = stable.exp()
    return exp_values / exp_values.sum(axis=axis, keepdims=True)


def gelu(x: Tensor) -> Tensor:
    coefficient = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + (coefficient * (x + 0.044715 * (x**3.0))).tanh())


def cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("cross_entropy expects logits with shape [batch, classes]")
    one_hot = np.eye(logits.shape[1], dtype=np.float32)[labels]
    shifted = logits - np.max(logits.data, axis=1, keepdims=True)
    log_prob = shifted - shifted.exp().sum(axis=1, keepdims=True).log()
    return -(log_prob * one_hot).sum(axis=1).mean()


class Adam:
    """Minimal Adam with global gradient clipping."""

    def __init__(
        self,
        parameters: Iterable[Tensor],
        *,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
    ) -> None:
        unique: dict[int, Tensor] = {id(p): p for p in parameters}
        self.parameters = list(unique.values())
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.step_count = 0

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()

    def step(self) -> float:
        self.step_count += 1
        squared_norm = sum(float(np.sum(p.grad**2)) for p in self.parameters)
        grad_norm = float(np.sqrt(squared_norm))
        scale = min(1.0, self.max_grad_norm / (grad_norm + 1e-12))

        beta1_correction = 1.0 - self.beta1**self.step_count
        beta2_correction = 1.0 - self.beta2**self.step_count
        for index, p in enumerate(self.parameters):
            grad = p.grad * scale
            self.m[index] = self.beta1 * self.m[index] + (1.0 - self.beta1) * grad
            self.v[index] = self.beta2 * self.v[index] + (1.0 - self.beta2) * (grad**2)
            m_hat = self.m[index] / beta1_correction
            v_hat = self.v[index] / beta2_correction
            p.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return grad_norm
