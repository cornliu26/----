"""Lesson 11 starter: GRU cell."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def init_gru_params(input_dim: int, hidden_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    scale = 0.01
    def rand():
        return rng.normal(scale=scale, size=(input_dim + hidden_dim, hidden_dim))
    return {
        "W_z": rand(),
        "W_r": rand(),
        "W_h": rand(),
        "b_z": np.zeros(hidden_dim),
        "b_r": np.zeros(hidden_dim),
        "b_h": np.zeros(hidden_dim),
    }


def gru_cell(x_t: np.ndarray, h_prev: np.ndarray, params: dict[str, np.ndarray]):
    """TODO(core): implement GRU update and return h_t."""
    raise NotImplementedError("Implement gru_cell.")


def quick_check() -> None:
    params = init_gru_params(input_dim=5, hidden_dim=7)
    x_t = np.random.randn(3, 5)
    h_prev = np.zeros((3, 7))
    print("CHECKPOINT: gru_cell should return shape (3, 7).")
    print("Param keys:", sorted(params))


if __name__ == "__main__":
    quick_check()
