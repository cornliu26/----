"""Loss functions for the tiny framework."""

from __future__ import annotations

import numpy as np


class MSELoss:
    def __init__(self) -> None:
        self.y_hat = None
        self.y_true = None

    def forward(self, y_hat: np.ndarray, y_true: np.ndarray) -> float:
        self.y_hat = y_hat
        self.y_true = y_true
        return float(np.mean((y_hat - y_true) ** 2) / 2.0)

    def backward(self) -> np.ndarray:
        return (self.y_hat - self.y_true) / self.y_true.shape[0]


class CrossEntropyLoss:
    def __init__(self) -> None:
        self.probs = None
        self.y_true = None
        self.eps = 1e-12

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.y_true = y_true

        correct_class_probs = self.probs[np.arange(y_true.shape[0]), y_true]
        return float(-np.mean(np.log(correct_class_probs + self.eps)))

    def backward(self) -> np.ndarray:
        n = self.y_true.shape[0]
        grand_logits = self.probs.copy()
        grand_logits[np.arange(n), self.y_true] -= 1.0
        return grand_logits / n
