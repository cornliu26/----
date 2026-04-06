"""Lesson 01 starter: numerical gradients."""

from __future__ import annotations

import math


def numerical_grad_scalar(f, x: float, eps: float = 1e-5) -> float:
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def numerical_grad_vector(f, values, eps: float = 1e-5):
    return [numerical_grad_scalar(f, x, eps) for x in values]


def quick_check() -> None:
    print("f(x)=x^2 at x=3 -> expected grad about 6")
    print("numerical:", numerical_grad_scalar(lambda x: x * x, 3.0))
    print("f(x)=sin(x) at x=0 -> expected grad about 1")
    print("numerical:", numerical_grad_scalar(math.sin, 0.0))
    print("CHECKPOINT: implement numerical_grad_vector for multivariable functions.")


if __name__ == "__main__":
    quick_check()
