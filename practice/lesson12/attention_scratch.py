"""Lesson 12 starter: scaled dot-product attention."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
):
    """Return (output, attention_weights) for scaled dot-product attention.

    Supported shapes:
    - Q: (num_queries, d_k)
    - K: (num_keys, d_k)
    - V: (num_keys, d_v)
    - mask: optional boolean array with shape (num_queries, num_keys)
      True means "keep", False means "mask out".
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)

    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K, and V must all be 2D arrays.")
    if K.shape[0] != V.shape[0]:
        raise ValueError("K and V must have the same number of rows.")
    if Q.shape[1] != K.shape[1]:
        raise ValueError("Q and K must share the same key dimension.")

    d_k = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(d_k)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != scores.shape:
            raise ValueError("mask must have the same shape as the score matrix.")
        scores = np.where(mask, scores, -1e9)

    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ V
    return output, attention_weights


def self_attention(
    X: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    mask: np.ndarray | None = None,
):
    """Project a token sequence into Q/K/V and apply self-attention."""
    X = np.asarray(X, dtype=float)
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    cache = {"Q": Q, "K": K, "V": V}
    return output, attention_weights, cache


def causal_mask(length: int) -> np.ndarray:
    """Lower-triangular mask for autoregressive self-attention."""
    if length <= 0:
        raise ValueError("length must be positive.")
    return np.tril(np.ones((length, length), dtype=bool))


def positional_encoding(length: int, d_model: int) -> np.ndarray:
    """Classic sinusoidal positional encoding."""
    if length <= 0 or d_model <= 0:
        raise ValueError("length and d_model must be positive.")

    positions = np.arange(length, dtype=float)[:, None]
    dims = np.arange(d_model, dtype=float)[None, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (dims // 2)) / d_model)
    angles = positions * angle_rates

    encodings = np.empty((length, d_model), dtype=float)
    encodings[:, 0::2] = np.sin(angles[:, 0::2])
    encodings[:, 1::2] = np.cos(angles[:, 1::2])
    return encodings


def pretty_print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"{name} shape: {matrix.shape}")
    print(np.array2string(matrix, precision=4, suppress_small=True))
    print()


def quick_check() -> None:
    np.set_printoptions(precision=4, suppress=True)

    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 20.0]])
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    pretty_print_matrix("Attention output", output)
    pretty_print_matrix("Attention weights", attention_weights)
    print("Weight row sums:", attention_weights.sum(axis=1))
    print("CHECKPOINT: each output row is a weighted sum of rows from V.")
    print()

    X = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    W_q = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    W_k = np.array(
        [
            [0.5, 0.0],
            [0.0, 1.0],
            [1.0, 0.5],
        ]
    )
    W_v = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    self_out, self_weights, cache = self_attention(X, W_q, W_k, W_v)
    pretty_print_matrix("Self-attention Q", cache["Q"])
    pretty_print_matrix("Self-attention weights", self_weights)
    pretty_print_matrix("Self-attention output", self_out)

    mask = causal_mask(length=X.shape[0])
    masked_out, masked_weights, _ = self_attention(X, W_q, W_k, W_v, mask=mask)
    pretty_print_matrix("Causal mask", mask.astype(int))
    pretty_print_matrix("Masked attention weights", masked_weights)
    pretty_print_matrix("Masked attention output", masked_out)

    pos = positional_encoding(length=4, d_model=6)
    pretty_print_matrix("Positional encoding", pos)
    print(
        "CHECKPOINT: self-attention lets every token look at other tokens; "
        "a causal mask blocks future tokens; positional encoding adds order information."
    )


if __name__ == "__main__":
    quick_check()
