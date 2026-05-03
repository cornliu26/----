"""Lesson 10 starter: a minimal RNN."""

from __future__ import annotations

import numpy as np


def init_rnn_params(vocab_size: int, hidden_size: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    scale = 0.01
    return {
        "W_xh": rng.normal(scale=scale, size=(vocab_size, hidden_size)),
        "W_hh": rng.normal(scale=scale, size=(hidden_size, hidden_size)),
        "b_h": np.zeros(hidden_size),
        "W_hq": rng.normal(scale=scale, size=(hidden_size, vocab_size)),
        "b_q": np.zeros(vocab_size),
    }


def one_hot(indices: np.ndarray, vocab_size: int) -> np.ndarray:
    out = np.zeros((indices.shape[0], vocab_size), dtype=float)
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def rnn_step(x_t: np.ndarray, h_prev: np.ndarray, params: dict[str, np.ndarray]):
    """Run one vanilla RNN step.

    x_t shape:     (batch_size, vocab_size)
    h_prev shape:  (batch_size, hidden_size)
    h_t shape:     (batch_size, hidden_size)
    logits_t shape:(batch_size, vocab_size)
    """
    hidden_pre_activation = (
        x_t @ params["W_xh"] + h_prev @ params["W_hh"] + params["b_h"]
    )
    h_t = np.tanh(hidden_pre_activation)
    logits_t = h_t @ params["W_hq"] + params["b_q"]
    return h_t, logits_t


def forward_sequence(
    input_indices: np.ndarray,
    params: dict[str, np.ndarray],
    vocab_size: int,
    h0: np.ndarray | None = None,
):
    """Unroll the RNN over a token sequence.

    input_indices shape: (time_steps,)
    Returns:
        hidden_states: (time_steps, 1, hidden_size)
        logits_seq:    (time_steps, 1, vocab_size)
        h_t:           final hidden state, shape (1, hidden_size)
    """
    input_indices = np.asarray(input_indices, dtype=int)
    hidden_size = params["W_hh"].shape[0]
    h_t = np.zeros((1, hidden_size), dtype=float) if h0 is None else h0.astype(float).copy()

    hidden_states = []
    logits_seq = []
    for idx in input_indices:
        x_t = one_hot(np.array([idx]), vocab_size)
        h_t, logits_t = rnn_step(x_t, h_t, params)
        hidden_states.append(h_t.copy())
        logits_seq.append(logits_t.copy())
    return np.stack(hidden_states), np.stack(logits_seq), h_t


def sample(
    start_idx: int,
    length: int,
    params: dict[str, np.ndarray],
    vocab_size: int,
    temperature: float = 1.0,
    seed: int = 0,
):
    """Generate token ids by autoregressive sampling."""
    if length <= 0:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    rng = np.random.default_rng(seed)
    hidden_size = params["W_hh"].shape[0]
    h_t = np.zeros((1, hidden_size), dtype=float)
    current_idx = int(start_idx)
    generated = [current_idx]

    for _ in range(length - 1):
        x_t = one_hot(np.array([current_idx]), vocab_size)
        h_t, logits_t = rnn_step(x_t, h_t, params)
        probs = softmax(logits_t / temperature)[0]
        current_idx = int(rng.choice(vocab_size, p=probs))
        generated.append(current_idx)
    return generated


def sequence_cross_entropy(logits_seq: np.ndarray, targets: np.ndarray) -> float:
    """Average next-token cross entropy over a sequence."""
    logits_seq = np.asarray(logits_seq, dtype=float)
    targets = np.asarray(targets, dtype=int)
    if logits_seq.shape[0] != targets.shape[0]:
        raise ValueError("logits_seq and targets must have the same time length.")

    probs = softmax(logits_seq[:, 0, :])
    correct = probs[np.arange(targets.shape[0]), targets]
    return float(-np.mean(np.log(correct + 1e-12)))


def quick_check() -> None:
    vocab_size = 8
    hidden_size = 16
    params = init_rnn_params(vocab_size=vocab_size, hidden_size=hidden_size)
    x_t = one_hot(np.array([1]), vocab_size)
    h_prev = np.zeros((1, hidden_size))
    h_t, logits_t = rnn_step(x_t, h_prev, params)
    toy_sequence = np.array([1, 2, 3, 4], dtype=int)
    hidden_states, logits_seq, _ = forward_sequence(toy_sequence, params, vocab_size)
    loss = sequence_cross_entropy(logits_seq, np.array([2, 3, 4, 5]) % vocab_size)
    sampled = sample(start_idx=1, length=6, params=params, vocab_size=vocab_size, seed=0)
    print("Parameter shapes:", {k: v.shape for k, v in params.items()})
    print("Input shape:", x_t.shape)
    print("Step output shapes:", h_t.shape, logits_t.shape)
    print("Sequence output shapes:", hidden_states.shape, logits_seq.shape)
    print("Toy sequence loss:", f"{loss:.4f}")
    print("Sampled token ids:", sampled)
    print(
        "CHECKPOINT: rnn_step updates the hidden state, forward_sequence unrolls over time, "
        "and sample feeds each predicted token back into the next step."
    )


if __name__ == "__main__":
    quick_check()
