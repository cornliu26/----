"""Lesson 10 starter: text preprocessing."""

from __future__ import annotations

from collections import Counter


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
    """TODO(core): build (x, y) subsequences for next-token prediction."""
    raise NotImplementedError("Implement make_subsequences.")


def top_counts(text: str, n: int = 10):
    return Counter(text).most_common(n)


def quick_check() -> None:
    text = clean_text("Deep Learning with Python")
    stoi, itos = build_char_vocab(text)
    indices = encode(text, stoi)
    print("Clean text:", text)
    print("Top counts:", top_counts(text))
    print("Decoded  :", decode(indices, itos))
    print("CHECKPOINT: implement make_subsequences.")


if __name__ == "__main__":
    quick_check()
