"""Lesson 07 starter: parameter save/load."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mydl.layers import Linear, ReLU, Sequential


def save_parameters(model, path: str) -> None:
    """TODO(core): save model parameters with numpy."""
    raise NotImplementedError("Implement save_parameters.")


def load_parameters(model, path: str) -> None:
    """TODO(core): load model parameters with numpy."""
    raise NotImplementedError("Implement load_parameters.")


def main() -> None:
    model = Sequential(Linear(2, 8), ReLU(), Linear(8, 3))
    print("CHECKPOINT: after implementation, save then load without changing parameter values.")
    print("Parameter shapes:", [param.data.shape for param in model.parameters()])


if __name__ == "__main__":
    main()
