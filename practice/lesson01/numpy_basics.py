"""Lesson 01 starter: NumPy basics.

Finish the TODO(core) items by editing this file directly.
"""

from __future__ import annotations

import numpy as np


def vector_dot(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size:
        raise ValueError("a and b must have the same size.")
    ret = 0.0
    for i in range(a.size):
        ret += a[i] * b[i]
    return ret


def matrix_vector_product(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    if A.shape[1] != x.size:
        raise ValueError("A must have the same number of columns as x.")
    ret = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ret[i] += A[i, j] * x[j]
    return ret


def matrix_matrix_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape[1] != B.shape[0]:
        raise ValueError("A must have the same number of columns as B.")
    ret = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                ret[i, j] += A[i, k] * B[k, j]
    return ret


def summarize_array(x: np.ndarray) -> dict[str, object]:
    return {
        "shape": x.shape,
        "ndim": x.ndim,
        "size": x.size,
        "dtype": str(x.dtype),
        "itemsize": x.itemsize,
        "nbytes": x.nbytes,
        "strides": x.strides,
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
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
    print("CHECKPOINT: vector_dot([1,2,3], [4,5,6]) should be 32.", vector_dot(a, b))
    print("CHECKPOINT: matrix_vector_product([[1,2],[3,4]], [10,20]) should be [50, 110].", matrix_vector_product(A, x))
    print("CHECKPOINT: matrix_matrix_product(A, B) should match A @ B.", matrix_matrix_product(A, B))


if __name__ == "__main__":
    quick_check()
