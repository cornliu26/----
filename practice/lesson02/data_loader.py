"""Lesson 02 starter: a tiny array dataloader."""

from __future__ import annotations

import numpy as np


def make_toy_regression_data(num_samples: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(num_samples, 2))
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 1.0
    return X, y


class ArrayDataLoader:
    def __init__(self, X, y, batch_size: int = 4, shuffle: bool = True):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        """TODO(core): yield mini-batches of (X_batch, y_batch)."""
        raise NotImplementedError("Implement __iter__.")


def quick_check() -> None:
    X, y = make_toy_regression_data()
    loader = ArrayDataLoader(X, y, batch_size=5, shuffle=False)
    print("CHECKPOINT: your loader should return 4 batches for 16 samples.")
    try:
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            print(f"batch {batch_idx}: X{X_batch.shape}, y{y_batch.shape}")
    except NotImplementedError as exc:
        print("Next step:", exc)


if __name__ == "__main__":
    quick_check()
