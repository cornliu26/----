"""Lesson 10 starter: a minimal RNN."""

from __future__ import annotations

import numpy as np


def init_rnn_params(vocab_size: int, hidden_size: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    scale = 0.01
    return {
        "W_xh": rng.normal(scale=scale, size=(vocab_size, hidden_size)),
        "W_hh": rng.normal(scale=scale, size=(hidden_size, hidden_size)),
        "b_h": np.zeros(hidden_size),
        "W_hq": rng.normal(scale=scale, size=(hidden_size, vocab_size)),
        "b_q": np.zeros(vocab_size),
    }


def one_hot(indices: np.ndarray, vocab_size: int) -> np.ndarray:
    out = np.zeros((indices.shape[0], vocab_size), dtype=float)
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out


def rnn_step(x_t: np.ndarray, h_prev: np.ndarray, params: dict[str, np.ndarray]):
    """TODO(core): implement one RNN step and return h_t, logits_t."""
    raise NotImplementedError("Implement rnn_step.")


def sample(start_idx: int, length: int, params: dict[str, np.ndarray], vocab_size: int):
    """TODO(core): generate indices from the model."""
    raise NotImplementedError("Implement sample.")


def quick_check() -> None:
    vocab_size = 8
    hidden_size = 16
    params = init_rnn_params(vocab_size=vocab_size, hidden_size=hidden_size)
    x_t = one_hot(np.array([1]), vocab_size)
    h_prev = np.zeros((1, hidden_size))
    print("Parameter shapes:", {k: v.shape for k, v in params.items()})
    print("Input shape:", x_t.shape)
    print("CHECKPOINT: rnn_step should return hidden shape (1, hidden_size) and logits shape (1, vocab_size).")


if __name__ == "__main__":
    quick_check()
