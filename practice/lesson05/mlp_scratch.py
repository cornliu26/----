"""Lesson 05 starter: MLP and backprop."""

from __future__ import annotations

import numpy as np


def init_params(input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(scale=0.01, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "W2": rng.normal(scale=0.01, size=(hidden_dim, output_dim)),
        "b2": np.zeros(output_dim),
    }
    return params


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    """TODO(core): implement stable softmax."""
    raise NotImplementedError("Implement softmax.")


def cross_entropy(probs: np.ndarray, y_true: np.ndarray) -> float:
    """TODO(core): implement average cross-entropy loss."""
    raise NotImplementedError("Implement cross_entropy.")


def forward_pass(X: np.ndarray, params: dict[str, np.ndarray]):
    """TODO(core): return probs and cache for backprop."""
    raise NotImplementedError("Implement forward_pass.")


def backward_pass(X: np.ndarray, y_true: np.ndarray, params: dict[str, np.ndarray], cache):
    """TODO(core): return gradients for W1, b1, W2, b2."""
    raise NotImplementedError("Implement backward_pass.")


def update_params(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float) -> None:
    for name in params:
        params[name] -= lr * grads[name]


def quick_check() -> None:
    X = np.random.randn(8, 4)
    y = np.random.randint(0, 3, size=8)
    params = init_params(input_dim=4, hidden_dim=5, output_dim=3)
    print("CHECKPOINT: forward_pass should output probs with shape (8, 3).")
    print("CHECKPOINT: backward_pass should return grads with shapes matching params.")
    print("Parameter shapes:", {k: v.shape for k, v in params.items()})


if __name__ == "__main__":
    quick_check()
