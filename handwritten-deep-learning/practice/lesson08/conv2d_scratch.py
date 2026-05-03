"""Lesson 08: 2D convolution from scratch.

In deep learning code, "convolution" is usually implemented as cross-correlation:
we slide the kernel over the input without flipping it.
"""

from __future__ import annotations

import numpy as np


def pad2d(x: np.ndarray, padding: int) -> np.ndarray:
    if padding < 0:
        raise ValueError("padding must be non-negative.")
    if padding == 0:
        return x
    return np.pad(x, ((padding, padding), (padding, padding)), mode="constant")


def compute_output_shape(
    input_shape: tuple[int, int],
    kernel_shape: tuple[int, int],
    stride: int = 1,
    padding: int = 0,
) -> tuple[int, int]:
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")
    in_h, in_w = input_shape
    k_h, k_w = kernel_shape
    padded_h = in_h + 2 * padding
    padded_w = in_w + 2 * padding
    if k_h > padded_h or k_w > padded_w:
        raise ValueError("kernel cannot be larger than the padded input.")
    out_h = (padded_h - k_h) // stride + 1
    out_w = (padded_w - k_w) // stride + 1
    return out_h, out_w


def conv2d_single_channel(
    x: np.ndarray,
    kernel: np.ndarray,
    bias: float = 0.0,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Apply a single 2D kernel to a single-channel image."""
    x = np.asarray(x, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")
    if kernel.ndim != 2:
        raise ValueError("kernel must be a 2D array.")

    x_padded = pad2d(x, padding)
    out_h, out_w = compute_output_shape(x.shape, kernel.shape, stride=stride, padding=padding)
    output = np.zeros((out_h, out_w), dtype=float)

    for out_i in range(out_h):
        for out_j in range(out_w):
            row_start = out_i * stride
            col_start = out_j * stride
            window = x_padded[row_start : row_start + kernel.shape[0], col_start : col_start + kernel.shape[1]]
            output[out_i, out_j] = np.sum(window * kernel) + bias

    return output


def quick_check() -> None:
    x = np.arange(1, 10, dtype=float).reshape(3, 3)
    kernel = np.array([[1.0, 0.0], [0.0, -1.0]])

    out = conv2d_single_channel(x, kernel, stride=1, padding=0)
    out_with_padding = conv2d_single_channel(x, kernel, stride=1, padding=1)
    out_with_stride = conv2d_single_channel(x, kernel, stride=2, padding=0)

    print("Input:\n", x)
    print("Kernel:\n", kernel)
    print("Output (stride=1, padding=0):\n", out)
    print("Output shape should be (2, 2):", out.shape)
    print("Output with padding=1 shape should be (4, 4):", out_with_padding.shape)
    print("Output with stride=2 shape should be (1, 1):", out_with_stride.shape)
    print("Expected first output matrix is all -4s:", np.allclose(out, -4.0 * np.ones((2, 2))))


if __name__ == "__main__":
    quick_check()
