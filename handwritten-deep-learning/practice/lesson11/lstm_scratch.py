"""Lesson 11 starter: LSTM cell."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def init_lstm_params(input_dim: int, hidden_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    scale = 0.01

    def rand():
        return rng.normal(scale=scale, size=(input_dim + hidden_dim, hidden_dim))

    return {
        "W_f": rand(),
        "W_i": rand(),
        "W_o": rand(),
        "W_g": rand(),
        "b_f": np.zeros(hidden_dim),
        "b_i": np.zeros(hidden_dim),
        "b_o": np.zeros(hidden_dim),
        "b_g": np.zeros(hidden_dim),
    }


def _concat_input_hidden(x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    return np.concatenate([x_t, h_prev], axis=1)


def lstm_step_details(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    params: dict[str, np.ndarray],
):
    """Run one LSTM step and expose all gate values."""
    combined = _concat_input_hidden(x_t, h_prev)
    f_t = sigmoid(combined @ params["W_f"] + params["b_f"])
    i_t = sigmoid(combined @ params["W_i"] + params["b_i"])
    o_t = sigmoid(combined @ params["W_o"] + params["b_o"])
    g_t = np.tanh(combined @ params["W_g"] + params["b_g"])

    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * np.tanh(c_t)
    cache = {
        "f_t": f_t,
        "i_t": i_t,
        "o_t": o_t,
        "g_t": g_t,
    }
    return h_t, c_t, cache


def lstm_cell(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    params: dict[str, np.ndarray],
):
    """Implement one LSTM step and return h_t, c_t.

    x_t shape:    (batch_size, input_dim)
    h_prev shape: (batch_size, hidden_dim)
    c_prev shape: (batch_size, hidden_dim)
    """
    h_t, c_t, _ = lstm_step_details(x_t, h_prev, c_prev, params)
    return h_t, c_t


def forward_lstm_sequence(
    inputs: np.ndarray,
    params: dict[str, np.ndarray],
    h0: np.ndarray | None = None,
    c0: np.ndarray | None = None,
):
    """Unroll an LSTM over a full sequence.

    inputs shape: (time_steps, batch_size, input_dim)
    returns:
        hidden_states shape: (time_steps, batch_size, hidden_dim)
        cell_states shape:   (time_steps, batch_size, hidden_dim)
        h_t, c_t: final states
    """
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (time_steps, batch_size, input_dim).")

    _, batch_size, _ = inputs.shape
    hidden_dim = params["b_f"].shape[0]
    h_t = np.zeros((batch_size, hidden_dim), dtype=float) if h0 is None else h0.astype(float).copy()
    c_t = np.zeros((batch_size, hidden_dim), dtype=float) if c0 is None else c0.astype(float).copy()

    hidden_states = []
    cell_states = []
    for x_t in inputs:
        h_t, c_t = lstm_cell(x_t, h_t, c_t, params)
        hidden_states.append(h_t.copy())
        cell_states.append(c_t.copy())
    return np.stack(hidden_states), np.stack(cell_states), h_t, c_t


def quick_check() -> None:
    rng = np.random.default_rng(0)
    params = init_lstm_params(input_dim=5, hidden_dim=7)
    x_t = rng.normal(size=(3, 5))
    h_prev = np.zeros((3, 7))
    c_prev = np.zeros((3, 7))
    h_t, c_t, gates = lstm_step_details(x_t, h_prev, c_prev, params)

    sequence = rng.normal(size=(4, 3, 5))
    hidden_states, cell_states, last_hidden, last_cell = forward_lstm_sequence(sequence, params)

    print("Single-step state shapes:", h_t.shape, c_t.shape)
    print("Gate shapes:", {name: value.shape for name, value in gates.items()})
    print(
        "Gate means:",
        {name: float(np.mean(value)) for name, value in gates.items()},
    )
    print(
        "Sequence state shapes:",
        hidden_states.shape,
        cell_states.shape,
        last_hidden.shape,
        last_cell.shape,
    )
    print("Param keys:", sorted(params))
    print(
        "CHECKPOINT: f_t controls forgetting, i_t controls writing, o_t controls how much "
        "of the cell state is exposed as the hidden state."
    )


if __name__ == "__main__":
    quick_check()
