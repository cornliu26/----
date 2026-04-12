"""Lesson 05 starter: MLP and backprop."""

from __future__ import annotations

import numpy as np


def init_params(input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(scale=0.01, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "W2": rng.normal(scale=0.01, size=(hidden_dim, output_dim)),
        "b2": np.zeros(output_dim),
    }
    return params


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs


def cross_entropy(probs: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-12
    n = probs.shape[0]
    correct_class_probs = probs[np.arange(n), y_true]
    return float(-np.mean(np.log(correct_class_probs + eps)))


def forward_pass(X: np.ndarray, params: dict[str, np.ndarray]):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    hidden_pre = X @ W1 + b1
    hidden = relu(hidden_pre)
    logits = hidden @ W2 + b2
    probs = softmax(logits)

    cache = {
        "X": X,
        "hidden_pre": hidden_pre,
        "hidden": hidden,
        "probs": probs,
    }
    return probs, cache


def backward_pass(X: np.ndarray, y_true: np.ndarray, params: dict[str, np.ndarray], cache):
    n = X.shape[0]

    W2 = params["W2"]

    probs = cache["probs"]
    hidden = cache["hidden"]
    hidden_pre = cache["hidden_pre"]
    X = cache["X"]

    grad_logits = probs.copy()
    grad_logits[np.arange(n), y_true] -= 1.0

    grad_W2 = hidden.T @ grad_logits / n
    grad_b2 = np.mean(grad_logits, axis=0)

    grad_hidden = grad_logits @ W2.T
    grad_hidden_pre = grad_hidden * (hidden_pre > 0)

    grad_W1 = X.T @ grad_hidden_pre / n
    grad_b1 = np.mean(grad_hidden_pre, axis=0)

    grads = {
        "W1": grad_W1,
        "b1": grad_b1,
        "W2": grad_W2,
        "b2": grad_b2,
    }
    return grads


def update_params(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float) -> None:
    for name in params:
        params[name] -= lr * grads[name]


def accuracy(logits: np.ndarray, y_true: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y_true))


def quick_check() -> None:
    num_epochs = 5000
    lr = 0.1
    num_samples = 800
    input_dim = 4
    hidden_dim = 5
    output_dim = 3

    X = np.random.randn(num_samples, input_dim)
    y = np.random.randint(0, 3, size=num_samples)
    params = init_params(input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    for epoch in range(num_epochs):
        probs, cache = forward_pass(X, params)
        loss = cross_entropy(probs, y)
        acc = accuracy(probs, y)

        grads = backward_pass(X, y, params, cache)
        update_params(params, grads, lr)

        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    print("CHECKPOINT: forward_pass should output probs with shape (8, 3).")
    print("CHECKPOINT: backward_pass should return grads with shapes matching params.")
    print("Parameter shapes:", {k: v.shape for k, v in params.items()})


if __name__ == "__main__":
    quick_check()
