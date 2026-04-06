"""Lesson 09 starter: compare optimizers on a toy objective."""

from __future__ import annotations

import numpy as np


def objective(point: np.ndarray) -> float:
    x, y = point
    return 0.5 * (4.0 * x * x + y * y)


def grad_objective(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.array([4.0 * x, y], dtype=float)


def run_sgd(start: np.ndarray, lr: float = 0.1, steps: int = 20):
    point = start.astype(float).copy()
    history = [objective(point)]
    for _ in range(steps):
        point -= lr * grad_objective(point)
        history.append(objective(point))
    return np.array(history)


def run_momentum(start: np.ndarray, lr: float = 0.1, beta: float = 0.9, steps: int = 20):
    """TODO(core): implement momentum optimization on the toy objective."""
    raise NotImplementedError("Implement run_momentum.")


def run_adam(start: np.ndarray, lr: float = 0.1, steps: int = 20):
    """TODO(core): implement Adam on the toy objective."""
    raise NotImplementedError("Implement run_adam.")


def main() -> None:
    start = np.array([5.0, 5.0], dtype=float)
    print("SGD history:", run_sgd(start)[:5])
    print("CHECKPOINT: add momentum and Adam, then compare which one descends faster.")


if __name__ == "__main__":
    main()
