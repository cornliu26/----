"""Lesson 06 starter: regularization experiments."""

from __future__ import annotations

import numpy as np


def l2_penalty(weights: list[np.ndarray]) -> float:
    return float(np.sum([np.sum(w**2) for w in weights]) / 2.0)


def add_weight_decay(grads: list[np.ndarray], weights: list[np.ndarray], weight_decay: float):
    """TODO(core): return gradients after L2 weight decay is added."""
    for i, (grad, w) in enumerate(zip(grads, weights)):
        grads[i] += weight_decay * w
    return grads


def compare_initializations(input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 10):
    rng = np.random.default_rng(0)
    zero_init = np.zeros((input_dim, hidden_dim))
    small_random = rng.normal(scale=0.01, size=(input_dim, hidden_dim))
    xavier_like = rng.normal(scale=np.sqrt(2.0 / (input_dim + hidden_dim)), size=(input_dim, hidden_dim))
    return {
        "zero": zero_init,
        "small_random": small_random,
        "xavier_like": xavier_like,
    }


def quick_check() -> None:
    inits = compare_initializations()
    print("Init stats:")
    for name, values in inits.items():
        print(name, "mean=", float(values.mean()), "std=", float(values.std()))
    print("CHECKPOINT: implement l2_penalty and add_weight_decay.")


if __name__ == "__main__":
    quick_check()
