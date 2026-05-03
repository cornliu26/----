"""Lesson 02 bridge exercise: a more realistic manual backprop example.

This file is intentionally more advanced than grad_check.py, but still small
enough to hand-derive on paper.

Computation graph:
    z = x @ W + b
    h = tanh(z)
    diff = h - y
    L = 0.5 * sum(diff^2)

Learning goals:
1. see how reverse-mode autodiff maps to matrix code
2. manually compute gradients for x, W, b
3. verify them with numerical gradient checking

TODO(optional):
    extend this single-sample version to a batch version
"""

from __future__ import annotations

import numpy as np


def forward_loss(x: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray):
    z = x @ W + b
    h = np.tanh(z)
    diff = h - y
    loss = 0.5 * np.sum(diff**2)
    cache = {
        "x": x,
        "W": W,
        "b": b,
        "y": y,
        "z": z,
        "h": h,
        "diff": diff,
        "loss": loss,
    }
    return loss, cache


def backward_loss(cache: dict[str, np.ndarray | float]):
    x = cache["x"]
    W = cache["W"]
    z = cache["z"]
    h = cache["h"]
    diff = cache["diff"]

    grad_diff = diff
    grad_h = grad_diff
    grad_z = grad_h * (1.0 - h**2)
    grad_W = np.outer(x, grad_z)
    grad_b = grad_z.copy()
    grad_x = W @ grad_z

    grads = {
        "diff": grad_diff,
        "h": grad_h,
        "z": grad_z,
        "W": grad_W,
        "b": grad_b,
        "x": grad_x,
    }
    return grads


def numerical_grad_array(f, arr: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    grad = np.zeros_like(arr, dtype=float)
    for idx in np.ndindex(arr.shape):
        original = arr[idx]
        arr[idx] = original + eps
        loss_pos = f(arr)
        arr[idx] = original - eps
        loss_neg = f(arr)
        arr[idx] = original
        grad[idx] = (loss_pos - loss_neg) / (2 * eps)
    return grad


def build_example():
    x = np.array([0.6, -1.2, 0.3], dtype=float)
    W = np.array(
        [
            [0.2, -0.4],
            [0.7, 0.1],
            [-0.5, 0.3],
        ],
        dtype=float,
    )
    b = np.array([0.1, -0.2], dtype=float)
    y = np.array([0.25, -0.75], dtype=float)
    return x, W, b, y


def quick_check() -> None:
    x, W, b, y = build_example()
    loss, cache = forward_loss(x, W, b, y)
    grads = backward_loss(cache)

    print("=== Forward ===")
    print("z    =", cache["z"])
    print("h    =", cache["h"])
    print("diff =", cache["diff"])
    print("loss =", float(loss))

    print("\n=== Backward (manual) ===")
    print("dL/ddiff =", grads["diff"])
    print("dL/dh    =", grads["h"])
    print("dL/dz    =", grads["z"])
    print("dL/db    =", grads["b"])
    print("dL/dx    =", grads["x"])
    print("dL/dW    =\n", grads["W"])

    num_grad_x = numerical_grad_array(lambda x_var: forward_loss(x_var, W, b, y)[0], x.copy())
    num_grad_W = numerical_grad_array(lambda W_var: forward_loss(x, W_var, b, y)[0], W.copy())
    num_grad_b = numerical_grad_array(lambda b_var: forward_loss(x, W, b_var, y)[0], b.copy())

    print("\n=== Gradient Check ===")
    print("x match:", np.allclose(grads["x"], num_grad_x, atol=1e-5))
    print("W match:", np.allclose(grads["W"], num_grad_W, atol=1e-5))
    print("b match:", np.allclose(grads["b"], num_grad_b, atol=1e-5))

    print("\n=== Numerical grads ===")
    print("num dL/dx =", num_grad_x)
    print("num dL/dW =\n", num_grad_W)
    print("num dL/db =", num_grad_b)

    print("\nCHECKPOINT:")
    print("1. Why is dL/diff equal to diff?")
    print("2. Why does tanh contribute the factor (1 - h^2)?")
    print("3. Why is dL/dW an outer product of x and dL/dz?")


if __name__ == "__main__":
    quick_check()
