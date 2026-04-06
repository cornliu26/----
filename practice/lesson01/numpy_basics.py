"""Lesson 01 starter: NumPy basics.

Finish the TODO(core) items by editing this file directly.
"""

from __future__ import annotations

import numpy as np


def vector_dot(a: np.ndarray, b: np.ndarray) -> float:
    """TODO(core): implement vector dot product without np.dot."""
    raise NotImplementedError("Implement vector_dot.")


def matrix_vector_product(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """TODO(core): implement matrix-vector product without using @."""
    raise NotImplementedError("Implement matrix_vector_product.")


def matrix_matrix_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """TODO(core): implement matrix-matrix product with loops."""
    raise NotImplementedError("Implement matrix_matrix_product.")


def summarize_array(x: np.ndarray) -> dict[str, object]:
    return {
        "shape": x.shape,
        "ndim": x.ndim,
        "dtype": str(x.dtype),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def demo_broadcasting() -> np.ndarray:
    col = np.array([[1.0], [2.0], [3.0]])
    row = np.array([[10.0, 20.0, 30.0]])
    return col + row


def quick_check() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.array([10.0, 20.0])
    B = np.array([[1.0, 2.0], [3.0, 4.0]])
    print("Broadcasting demo:\n", demo_broadcasting())
    print("Array summary:", summarize_array(A))
    print("CHECKPOINT: vector_dot([1,2,3], [4,5,6]) should be 32.")
    print("CHECKPOINT: matrix_vector_product([[1,2],[3,4]], [10,20]) should be [50, 110].")
    print("CHECKPOINT: matrix_matrix_product(A, B) should match A @ B.")


if __name__ == "__main__":
    quick_check()
