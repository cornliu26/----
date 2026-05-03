"""Lesson 02 starter: gradient checking."""

from __future__ import annotations

import numpy as np


def numerical_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        original = x.flat[i]
        x.flat[i] = original + eps
        fx_pos = f(x)
        x.flat[i] = original - eps
        fx_neg = f(x)
        x.flat[i] = original
        grad.flat[i] = (fx_pos - fx_neg) / (2 * eps)
    return grad


def analytical_grad_example(x: np.ndarray) -> np.ndarray:
    return 2.0 * x + 3.0


def quick_check() -> None:
    x = np.array([1.0, -2.0, 3.0], dtype=float)

    def fn(values):
        return float(np.sum(values**2 + 3.0 * values))

    numerical = numerical_grad(fn, x.copy())
    analytical = analytical_grad_example(x)
    print("Numerical grad:", numerical)
    print("Analytical grad:", analytical)
    print("Match:", np.allclose(numerical, analytical))


if __name__ == "__main__":
    quick_check()
