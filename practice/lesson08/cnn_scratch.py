"""Lesson 08 starter: minimal CNN structure."""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def max_pool2d(x: np.ndarray, kernel_size: int = 2, stride: int = 2):
    """TODO(core): implement max pooling for a single sample single channel input."""
    raise NotImplementedError("Implement max_pool2d.")


def describe_pipeline() -> None:
    print("Suggested pipeline: Conv -> ReLU -> Pool -> Flatten -> Linear")
    print("CHECKPOINT: finish conv2d_scratch.py first, then extend this file.")


if __name__ == "__main__":
    describe_pipeline()
