"""Lesson 10 starter: text preprocessing."""

from __future__ import annotations

from collections import Counter

import numpy as np


def clean_text(text: str) -> str:
    return " ".join(text.lower().split())


def build_char_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for ch, idx in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]):
    return [stoi[ch] for ch in text]


def decode(indices, itos: dict[int, str]):
    return "".join(itos[idx] for idx in indices)


def make_subsequences(indices, num_steps: int):
    """Build sliding windows for next-token prediction.

    Example with ``num_steps=4``:
    text ids:  [1, 5, 2, 7, 3, 4]
    x sample:  [1, 5, 2, 7]
    y sample:  [5, 2, 7, 3]
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D sequence.")

    num_samples = indices.shape[0] - num_steps
    if num_samples <= 0:
        empty = np.empty((0, num_steps), dtype=int)
        return empty, empty.copy()

    X = np.empty((num_samples, num_steps), dtype=int)
    y = np.empty((num_samples, num_steps), dtype=int)
    for start in range(num_samples):
        X[start] = indices[start : start + num_steps]
        y[start] = indices[start + 1 : start + num_steps + 1]
    return X, y


def top_counts(text: str, n: int = 10):
    return Counter(text).most_common(n)


def quick_check() -> None:
    text = clean_text("Deep Learning with Python")
    stoi, itos = build_char_vocab(text)
    indices = encode(text, stoi)
    X, y = make_subsequences(indices, num_steps=5)
    print("Clean text:", text)
    print("Top counts:", top_counts(text))
    print("Decoded  :", decode(indices, itos))
    print("Subsequence shapes:", X.shape, y.shape)
    if len(X) > 0:
        print("First x sample:", X[0], "->", repr(decode(X[0], itos)))
        print("First y sample:", y[0], "->", repr(decode(y[0], itos)))
    print("CHECKPOINT: each y sample should be x shifted left by one token.")


if __name__ == "__main__":
    quick_check()
