"""Lesson 06 starter: dropout."""

from __future__ import annotations

import numpy as np


def dropout_forward(x: np.ndarray, drop_prob: float, training: bool, seed: int | None = None):
    """TODO(core): implement inverted dropout.

    Return:
        out: output after dropout
        mask: dropout mask used during training, or None during eval
    """
    raise NotImplementedError("Implement dropout_forward.")


def quick_check() -> None:
    x = np.ones((4, 4), dtype=float)
    print("CHECKPOINT: in eval mode, output should equal input.")
    print("CHECKPOINT: in train mode, some entries should become 0 and others should be scaled.")
    print("Input:\n", x)


if __name__ == "__main__":
    quick_check()
