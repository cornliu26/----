"""Utility helpers that are not the main learning target."""

from __future__ import annotations

import numpy as np


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)


def accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == y_true))
