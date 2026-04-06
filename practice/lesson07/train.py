"""Lesson 07 starter: train with the tiny framework."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mydl.layers import Linear, ReLU, Sequential
from mydl.losses import CrossEntropyLoss
from mydl.optim import SGD
from mydl.utils import accuracy_from_logits


def make_toy_data(samples_per_class: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = np.array([[2.0, 2.0], [-2.0, 2.0], [0.0, -2.0]])
    X_parts = []
    y_parts = []
    for class_id, center in enumerate(centers):
        X_parts.append(center + 0.6 * rng.normal(size=(samples_per_class, 2)))
        y_parts.append(np.full(samples_per_class, class_id))
    return np.vstack(X_parts), np.concatenate(y_parts)


def main() -> None:
    X, y = make_toy_data()
    model = Sequential(Linear(2, 16), ReLU(), Linear(16, 3))
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    print("CHECKPOINT: after you implement mydl pieces, this script can become your first end-to-end training loop.")
    print("Current model parameter count:", sum(p.data.size for p in model.parameters()))
    try:
        logits = model.forward(X[:4])
        print("Current logits shape for 4 samples:", logits.shape)
        loss = criterion.forward(logits, y[:4])
        print("Current loss:", loss)
        print("Current acc :", accuracy_from_logits(logits, y[:4]))
    except NotImplementedError as exc:
        print("Expected for now:", exc)
        print("Suggested order: Linear.forward -> CrossEntropyLoss.forward -> Linear.backward -> optimizer improvements.")
    optimizer.zero_grad()


if __name__ == "__main__":
    main()
