"""Lesson 04 starter: softmax regression."""

from __future__ import annotations

import numpy as np

from metrics import accuracy


def make_toy_classification_data(samples_per_class: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = np.array([[2.0, 2.0], [-2.0, 2.0], [0.0, -2.0]])
    X_parts = []
    y_parts = []
    for class_id, center in enumerate(centers):
        X_parts.append(center + 0.6 * rng.normal(size=(samples_per_class, 2)))
        y_parts.append(np.full(samples_per_class, class_id))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def linear_logits(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b


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


def gradients(X: np.ndarray, probs: np.ndarray, y_true: np.ndarray):
    n = X.shape[0]
    grand_logits = probs.copy()
    grand_logits[np.arange(n), y_true] -= 1.0
    grad_W = X.T @ grand_logits / n
    grad_b = np.mean(grand_logits, axis=0)
    return grad_W, grad_b


def train(num_epochs: int = 50, lr: float = 0.1):
    X, y = make_toy_classification_data()
    num_features = X.shape[1]
    num_classes = int(y.max()) + 1
    rng = np.random.default_rng(42)
    W = rng.normal(scale=0.01, size=(num_features, num_classes))
    b = np.zeros(num_classes)

    for epoch in range(num_epochs):
        logits = linear_logits(X, W, b)
        probs = softmax(logits)
        loss = cross_entropy(probs, y)
        grad_W, grad_b = gradients(X, probs, y)
        W -= lr * grad_W
        b -= lr * grad_b
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            logits = linear_logits(X, W, b)
            print(
                f"epoch={epoch:02d} "
                f"loss={loss:.4f} "
                f"acc={accuracy(logits, y):.4f}"
            )


if __name__ == "__main__":
    print("CHECKPOINT: after implementation, acc should rise well above random guess.")
    try:
        train(5000)
    except NotImplementedError as exc:
        print("Next step:", exc)
        print("Suggested order: softmax -> cross_entropy -> gradients -> train.")
