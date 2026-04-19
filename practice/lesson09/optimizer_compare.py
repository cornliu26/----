"""Lesson 09 starter: compare optimizers on a toy objective."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mydl.base import Parameter
from mydl.optim import Adam, Momentum, SGD


@dataclass
class RunResult:
    name: str
    history: np.ndarray
    trajectory: np.ndarray

    @property
    def final_loss(self) -> float:
        return float(self.history[-1])

    @property
    def final_point(self) -> np.ndarray:
        return self.trajectory[-1]


def objective(point: np.ndarray) -> float:
    x, y = point
    return 0.5 * (4.0 * x * x + y * y)


def grad_objective(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.array([4.0 * x, y], dtype=float)


def run_optimizer(
    name: str,
    optimizer_cls,
    start: np.ndarray,
    steps: int = 20,
    **optimizer_kwargs,
) -> RunResult:
    point = Parameter(start.astype(float).copy())
    history = np.empty(steps + 1, dtype=float)
    trajectory = np.empty((steps + 1, start.size), dtype=float)
    optimizer = optimizer_cls([point], **optimizer_kwargs)

    history[0] = objective(point.data)
    trajectory[0] = point.data

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        point.grad[...] = grad_objective(point.data)
        optimizer.step()

        history[step] = objective(point.data)
        trajectory[step] = point.data

    return RunResult(name=name, history=history, trajectory=trajectory)


def run_sgd(start: np.ndarray, lr: float = 0.1, steps: int = 20) -> RunResult:
    return run_optimizer("SGD", SGD, start=start, lr=lr, steps=steps)


def run_momentum(
    start: np.ndarray,
    lr: float = 0.1,
    beta: float = 0.9,
    steps: int = 20,
) -> RunResult:
    return run_optimizer("Momentum", Momentum, start=start, lr=lr, beta=beta, steps=steps)


def run_adam(start: np.ndarray, lr: float = 0.1, steps: int = 20) -> RunResult:
    return run_optimizer("Adam", Adam, start=start, lr=lr, steps=steps)


def first_step_below(history: np.ndarray, threshold: float) -> int | None:
    hit_steps = np.where(history <= threshold)[0]
    if hit_steps.size == 0:
        return None
    return int(hit_steps[0])


def format_result(result: RunResult, threshold: float = 1.0) -> str:
    step_hit = first_step_below(result.history, threshold)
    hit_text = "never" if step_hit is None else str(step_hit)
    return (
        f"{result.name:<8} "
        f"final_loss={result.final_loss:>9.6f} "
        f"final_point={np.array2string(result.final_point, precision=4)} "
        f"step(loss<={threshold})={hit_text}"
    )


def main() -> None:
    start = np.array([5.0, 5.0], dtype=float)
    steps = 20
    same_lr_results = [
        run_sgd(start, lr=0.1, steps=steps),
        run_momentum(start, lr=0.1, beta=0.9, steps=steps),
        run_adam(start, lr=0.1, steps=steps),
    ]
    tuned_results = [
        run_sgd(start, lr=0.1, steps=steps),
        run_momentum(start, lr=0.05, beta=0.7, steps=steps),
        run_adam(start, lr=0.3, steps=steps),
    ]

    np.set_printoptions(precision=4, suppress=True)
    print("Objective: f(x, y) = 0.5 * (4x^2 + y^2)")
    print("Start point:", start)
    print()
    print("Experiment 1: same learning rate")
    print("Config: SGD lr=0.1 | Momentum lr=0.1 beta=0.9 | Adam lr=0.1")
    for result in same_lr_results:
        print(format_result(result))
        print(f"  first 5 losses: {np.array2string(result.history[:5], precision=4)}")
    print(
        "  takeaway: the same nominal lr is not equally aggressive for every optimizer. "
        "Momentum can overshoot here, and Adam is quite conservative with lr=0.1."
    )
    print()
    print("Experiment 2: light hyperparameter tuning")
    print("Config: SGD lr=0.1 | Momentum lr=0.05 beta=0.7 | Adam lr=0.3")
    for result in tuned_results:
        print(format_result(result))
        print(f"  first 5 losses: {np.array2string(result.history[:5], precision=4)}")
    print()
    best = min(tuned_results, key=lambda result: result.final_loss)
    print("Lowest final loss after light tuning:", best.name)
    print(
        "CHECKPOINT: explain both experiments. First compare why equal lr can mislead; "
        "then explain why tuned Momentum or Adam can beat plain SGD on this anisotropic quadratic."
    )


if __name__ == "__main__":
    main()
