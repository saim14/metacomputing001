from __future__ import annotations

import numpy as np

from src.tiny_autograd import Tensor, cross_entropy, parameter, softmax


def finite_difference(function, values: np.ndarray, index: tuple[int, ...], epsilon: float = 1e-3) -> float:
    plus = values.copy()
    minus = values.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    return float((function(plus) - function(minus)) / (2.0 * epsilon))


def test_broadcast_matmul_gradient() -> None:
    rng = np.random.default_rng(3)
    x_data = rng.normal(size=(2, 3, 4)).astype(np.float32)
    w_data = rng.normal(size=(4, 2)).astype(np.float32)
    x = Tensor(x_data, requires_grad=True)
    w = parameter(w_data, "w")
    loss = ((x @ w) ** 2.0).mean()
    loss.backward()

    def objective(candidate: np.ndarray) -> float:
        return float(np.mean((x_data @ candidate) ** 2.0))

    numeric = finite_difference(objective, w_data, (1, 0))
    assert np.isclose(w.grad[1, 0], numeric, rtol=2e-2, atol=2e-3)


def test_softmax_rows_sum_to_one() -> None:
    logits = Tensor([[1.0, 2.0, -1.0], [0.2, 0.2, 0.2]], requires_grad=True)
    probabilities = softmax(logits)
    assert np.allclose(probabilities.data.sum(axis=1), 1.0)


def test_cross_entropy_gradient_sums_to_zero_per_row() -> None:
    logits = Tensor([[1.0, -0.5], [0.0, 1.0]], requires_grad=True)
    loss = cross_entropy(logits, np.array([0, 1]))
    loss.backward()
    assert np.allclose(logits.grad.sum(axis=1), 0.0, atol=1e-6)

