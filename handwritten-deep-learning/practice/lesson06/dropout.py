"""Lesson 06 starter: dropout."""

from __future__ import annotations

import numpy as np


def dropout_forward(x: np.ndarray, drop_prob: float, training: bool, seed: int | None = None):
    if not training:
        return x

    rng = np.random.default_rng(seed)
    mask = rng.binomial(1, 1 - drop_prob, size=x.shape)
    mask = mask / (1 - drop_prob)
    out = x * mask

    return out


def quick_check() -> None:
    x = np.ones((4, 4), dtype=float)
    print("CHECKPOINT: in eval mode, output should equal input.")
    print("CHECKPOINT: in train mode, some entries should become 0 and others should be scaled.")
    print("Input:\n", x)


if __name__ == "__main__":
    quick_check()
