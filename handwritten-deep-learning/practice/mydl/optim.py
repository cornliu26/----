"""Optimizers for the tiny framework."""

from __future__ import annotations

import numpy as np

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
        self.velocity = [np.zeros_like(param.data, dtype=float) for param in params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        for idx, param in enumerate(self.params):
            self.velocity[idx] = self.beta * self.velocity[idx] + param.grad
            param.data -= self.lr * self.velocity[idx]


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
        self.m = [np.zeros_like(param.data, dtype=float) for param in params]
        self.v = [np.zeros_like(param.data, dtype=float) for param in params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        self.t += 1
        for idx, param in enumerate(self.params):
            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * param.grad
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * (param.grad ** 2)

            m_hat = self.m[idx] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1.0 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
