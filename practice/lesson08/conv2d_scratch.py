"""Lesson 08 starter: 2D convolution."""

from __future__ import annotations

import numpy as np


def pad2d(x: np.ndarray, padding: int) -> np.ndarray:
    if padding == 0:
        return x
    return np.pad(x, ((padding, padding), (padding, padding)), mode="constant")


def conv2d_single_channel(x: np.ndarray, kernel: np.ndarray, bias: float = 0.0, stride: int = 1, padding: int = 0):
    """TODO(core): implement single-channel 2D convolution."""
    raise NotImplementedError("Implement conv2d_single_channel.")


def quick_check() -> None:
    x = np.arange(1, 10, dtype=float).reshape(3, 3)
    kernel = np.array([[1.0, 0.0], [0.0, -1.0]])
    print("Input:\n", x)
    print("Kernel:\n", kernel)
    print("CHECKPOINT: output with stride=1, padding=0 should have shape (2, 2).")


if __name__ == "__main__":
    quick_check()
