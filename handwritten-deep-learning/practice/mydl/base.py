"""Core building blocks for lesson 07 onward."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Parameter:
    data: np.ndarray
    grad: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=float)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class Module:
    def __init__(self) -> None:
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self) -> list[Parameter]:
        return []

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False
