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
    print("CHECKPOINT: this script now exercises a full training loop.")
    print("Current model parameter count:", sum(p.data.size for p in model.parameters()))

    rng = np.random.default_rng(0)
    batch_indices = rng.choice(X.shape[0], size=32, replace=False)
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]

    for epoch in range(20):
        optimizer.zero_grad()
        logits = model.forward(X_batch)
        loss = criterion.forward(logits, y_batch)
        grad_logits = criterion.backward()
        model.backward(grad_logits)
        optimizer.step()

        if epoch == 0 or epoch == 19 or epoch % 5 == 0:
            print(
                f"epoch={epoch:02d} "
                f"loss={loss:.4f} "
                f"acc={accuracy_from_logits(logits, y_batch):.4f}"
            )

    print("Final logits shape:", logits.shape)
    print("Gradient shapes:", [param.grad.shape for param in model.parameters()])


if __name__ == "__main__":
    main()
