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
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(self.X.shape[0])
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]    


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
