"""Optimizers for the tiny framework."""

from __future__ import annotations

from mydl.base import Parameter


class SGD:
    def __init__(self, params: list[Parameter], lr: float = 0.01) -> None:
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        for param in self.params:
            param.data -= self.lr * param.grad


class Momentum:
    def __init__(self, params: list[Parameter], lr: float = 0.01, beta: float = 0.9) -> None:
        self.params = params
        self.lr = lr
        self.beta = beta
        self.velocity = [0.0 for _ in params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        """TODO(core): implement momentum updates."""
        raise NotImplementedError("Implement Momentum.step.")


class Adam:
    def __init__(
        self,
        params: list[Parameter],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0 for _ in params]
        self.v = [0.0 for _ in params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        """TODO(core): implement Adam with bias correction."""
        raise NotImplementedError("Implement Adam.step.")
