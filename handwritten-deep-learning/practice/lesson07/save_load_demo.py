"""Lesson 07 starter: parameter save/load."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mydl.layers import Linear, ReLU, Sequential


def save_parameters(model, path: str) -> None:
    arrays = {
        f"param_{idx}": param.data for idx, param in enumerate(model.parameters())
    }
    np.savez(path, **arrays)


def load_parameters(model, path: str) -> None:
    params = np.load(path)
    for idx, param in enumerate(model.parameters()):
        key = f"param_{idx}"
        if key not in params:
            raise KeyError(f"Missing parameter {key} in saved file.")
        if params[key].shape != param.data.shape:
            raise ValueError(
                f"Shape mismatch for {key}: expected {param.data.shape}, got {params[key].shape}."
            )
        param.data[...] = params[key]


def main() -> None:
    model = Sequential(Linear(2, 8), ReLU(), Linear(8, 3))
    original_params = [param.data.copy() for param in model.parameters()]
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        save_path = tmp.name

    save_parameters(model, save_path)

    for param in model.parameters():
        param.data += 1.0

    load_parameters(model, save_path)
    restored_params = [param.data for param in model.parameters()]

    print("CHECKPOINT: save then load without changing parameter values.")
    print("Parameter shapes:", [param.data.shape for param in model.parameters()])
    print(
        "Restore match:",
        all(np.allclose(before, after) for before, after in zip(original_params, restored_params)),
    )


if __name__ == "__main__":
    main()
