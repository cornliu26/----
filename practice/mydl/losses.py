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

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """TODO(core): compute stable softmax probabilities and loss."""
        raise NotImplementedError("Implement CrossEntropyLoss.forward.")

    def backward(self) -> np.ndarray:
        """TODO(core): return gradient wrt logits."""
        raise NotImplementedError("Implement CrossEntropyLoss.backward.")
