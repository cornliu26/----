"""Lesson 11 starter: GRU cell."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def init_gru_params(input_dim: int, hidden_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    scale = 0.01

    def rand():
        return rng.normal(scale=scale, size=(input_dim + hidden_dim, hidden_dim))

    return {
        "W_z": rand(),
        "W_r": rand(),
        "W_h": rand(),
        "b_z": np.zeros(hidden_dim),
        "b_r": np.zeros(hidden_dim),
        "b_h": np.zeros(hidden_dim),
    }


def _concat_input_hidden(x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    return np.concatenate([x_t, h_prev], axis=1)


def gru_step_details(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    params: dict[str, np.ndarray],
):
    """Run one GRU step and also expose the gate values for inspection."""
    combined = _concat_input_hidden(x_t, h_prev)
    z_t = sigmoid(combined @ params["W_z"] + params["b_z"])
    r_t = sigmoid(combined @ params["W_r"] + params["b_r"])

    candidate_input = _concat_input_hidden(x_t, r_t * h_prev)
    h_tilde = np.tanh(candidate_input @ params["W_h"] + params["b_h"])

    h_t = z_t * h_prev + (1.0 - z_t) * h_tilde
    cache = {
        "z_t": z_t,
        "r_t": r_t,
        "h_tilde": h_tilde,
    }
    return h_t, cache


def gru_cell(x_t: np.ndarray, h_prev: np.ndarray, params: dict[str, np.ndarray]):
    """Implement one GRU step and return the new hidden state.

    x_t shape:    (batch_size, input_dim)
    h_prev shape: (batch_size, hidden_dim)
    h_t shape:    (batch_size, hidden_dim)
    """
    h_t, _ = gru_step_details(x_t, h_prev, params)
    return h_t


def forward_gru_sequence(
    inputs: np.ndarray,
    params: dict[str, np.ndarray],
    h0: np.ndarray | None = None,
):
    """Unroll a GRU over a full sequence.

    inputs shape: (time_steps, batch_size, input_dim)
    returns:
        hidden_states shape: (time_steps, batch_size, hidden_dim)
        h_t shape:           (batch_size, hidden_dim)
    """
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (time_steps, batch_size, input_dim).")

    _, batch_size, _ = inputs.shape
    hidden_dim = params["b_z"].shape[0]
    h_t = np.zeros((batch_size, hidden_dim), dtype=float) if h0 is None else h0.astype(float).copy()

    hidden_states = []
    for x_t in inputs:
        h_t = gru_cell(x_t, h_t, params)
        hidden_states.append(h_t.copy())
    return np.stack(hidden_states), h_t


def quick_check() -> None:
    rng = np.random.default_rng(0)
    params = init_gru_params(input_dim=5, hidden_dim=7)
    x_t = rng.normal(size=(3, 5))
    h_prev = np.zeros((3, 7))
    h_t, gates = gru_step_details(x_t, h_prev, params)

    sequence = rng.normal(size=(4, 3, 5))
    hidden_states, last_hidden = forward_gru_sequence(sequence, params)

    print("Single-step hidden shape:", h_t.shape)
    print("Gate shapes:", {name: value.shape for name, value in gates.items()})
    print(
        "Gate means:",
        {name: float(np.mean(value)) for name, value in gates.items()},
    )
    print("Sequence hidden shapes:", hidden_states.shape, last_hidden.shape)
    print("Param keys:", sorted(params))
    print(
        "CHECKPOINT: z_t decides how much old state to keep, r_t decides how much old "
        "state to expose when building the candidate hidden state."
    )


if __name__ == "__main__":
    quick_check()
