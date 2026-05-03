"""Lesson 03 starter: linear regression from scratch."""

from __future__ import annotations

import numpy as np


def generate_data(num_samples: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(num_samples, 2))
    w_true = np.array([2.0, -3.4])
    b_true = 4.2
    y = X @ w_true + b_true + rng.normal(scale=0.01, size=num_samples)
    return X, y, w_true, b_true


def model(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def mse_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * np.mean((y_hat - y) ** 2)

def compute_gradients(X: np.ndarray, y_hat: np.ndarray, y: np.ndarray):
    diff = y_hat - y
    grad_w = X.T @ diff / X.shape[0]
    grad_b = np.mean(diff)
    return grad_w, grad_b

def sgd_step(w: np.ndarray, b: float, grad_w: np.ndarray, grad_b: float, lr: float):
    return w - lr * grad_w, b - lr * grad_b


def train(num_epochs: int = 20, lr: float = 0.03):
    X, y, w_true, b_true = generate_data()
    rng = np.random.default_rng(42)
    w = rng.normal(scale=0.01, size=X.shape[1])
    b = 0.0
    history = []

    for epoch in range(num_epochs):
        y_hat = model(X, w, b)
        loss = mse_loss(y_hat, y)
        grad_w, grad_b = compute_gradients(X, y_hat, y)
        w, b = sgd_step(w, b, grad_w, grad_b, lr)
        history.append(loss)
        print(f"epoch={epoch:02d} loss={loss:.6f}")

    print("Learned w:", w)
    print("True w   :", w_true)
    print("Learned b:", b)
    print("True b   :", b_true)
    return history


if __name__ == "__main__":
    print("CHECKPOINT: after implementation, loss should go down steadily.")
    try:
        train(2000)
    except NotImplementedError as exc:
        print("Next step:", exc)
        print("Suggested order: model -> mse_loss -> compute_gradients -> train loop.")
