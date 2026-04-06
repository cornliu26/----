"""Lesson 11 starter: LSTM cell."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def init_lstm_params(input_dim: int, hidden_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    scale = 0.01
    def rand():
        return rng.normal(scale=scale, size=(input_dim + hidden_dim, hidden_dim))
    return {
        "W_f": rand(),
        "W_i": rand(),
        "W_o": rand(),
        "W_g": rand(),
        "b_f": np.zeros(hidden_dim),
        "b_i": np.zeros(hidden_dim),
        "b_o": np.zeros(hidden_dim),
        "b_g": np.zeros(hidden_dim),
    }


def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray, params: dict[str, np.ndarray]):
    """TODO(core): implement one LSTM step and return h_t, c_t."""
    raise NotImplementedError("Implement lstm_cell.")


def quick_check() -> None:
    params = init_lstm_params(input_dim=5, hidden_dim=7)
    x_t = np.random.randn(3, 5)
    h_prev = np.zeros((3, 7))
    c_prev = np.zeros((3, 7))
    print("CHECKPOINT: lstm_cell should return two arrays with shape (3, 7).")
    print("Param keys:", sorted(params))


if __name__ == "__main__":
    quick_check()
