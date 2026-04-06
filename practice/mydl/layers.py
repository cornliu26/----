"""Educational layers for the tiny framework."""

from __future__ import annotations

import numpy as np

from mydl.base import Module, Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, seed: int = 0) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Parameter(rng.normal(scale=scale, size=(in_features, out_features)))
        self.bias = Parameter(np.zeros(out_features))
        self.last_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """TODO(core): store input and compute x @ W + b."""
        raise NotImplementedError("Implement Linear.forward.")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """TODO(core): accumulate grads and return grad wrt input."""
        raise NotImplementedError("Implement Linear.backward.")

    def parameters(self) -> list[Parameter]:
        return [self.weight, self.bias]


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_input = x
        return np.maximum(x, 0.0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.last_input > 0)


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self) -> list[Parameter]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
