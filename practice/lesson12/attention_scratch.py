"""Lesson 12 starter: scaled dot-product attention."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """TODO(core): return (output, attention_weights)."""
    raise NotImplementedError("Implement scaled_dot_product_attention.")


def quick_check() -> None:
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 20.0]])
    print("Q shape:", Q.shape, "K shape:", K.shape, "V shape:", V.shape)
    print("CHECKPOINT: the output should be a weighted sum of rows from V.")
    print("CHECKPOINT: attention weights should sum to 1 across keys.")


if __name__ == "__main__":
    quick_check()
