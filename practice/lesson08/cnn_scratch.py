"""Lesson 08: a tiny CNN-style pipeline.

This file focuses on the forward pipeline:
    Conv -> ReLU -> Pool -> Flatten -> Linear

To keep the code readable for lesson08, we only train the final linear head.
The convolution filters are fixed by hand so you can focus on data flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from conv2d_scratch import conv2d_single_channel


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def max_pool2d(x: np.ndarray, kernel_size: int = 2, stride: int = 2) -> np.ndarray:
    """Max pooling for a single sample single-channel input."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")
    if kernel_size <= 0 or stride <= 0:
        raise ValueError("kernel_size and stride must be positive integers.")
    if kernel_size > x.shape[0] or kernel_size > x.shape[1]:
        raise ValueError("kernel_size cannot be larger than input dimensions.")

    out_h = (x.shape[0] - kernel_size) // stride + 1
    out_w = (x.shape[1] - kernel_size) // stride + 1
    output = np.zeros((out_h, out_w), dtype=float)

    for out_i in range(out_h):
        for out_j in range(out_w):
            row_start = out_i * stride
            col_start = out_j * stride
            window = x[row_start : row_start + kernel_size, col_start : col_start + kernel_size]
            output[out_i, out_j] = np.max(window)

    return output


def apply_filter_bank(
    image: np.ndarray,
    kernels: np.ndarray,
    biases: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Apply multiple kernels to one single-channel image."""
    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")
    if kernels.ndim != 3:
        raise ValueError("kernels must have shape (num_kernels, k_h, k_w).")

    num_kernels = kernels.shape[0]
    if biases is None:
        biases = np.zeros(num_kernels, dtype=float)
    feature_maps = []
    for kernel_idx in range(num_kernels):
        feature_map = conv2d_single_channel(
            image,
            kernels[kernel_idx],
            bias=float(biases[kernel_idx]),
            stride=stride,
            padding=padding,
        )
        feature_maps.append(feature_map)
    return np.stack(feature_maps, axis=0)


def cnn_feature_extractor(
    images: np.ndarray,
    kernels: np.ndarray,
    biases: np.ndarray | None = None,
    conv_stride: int = 1,
    conv_padding: int = 0,
    pool_kernel_size: int = 2,
    pool_stride: int = 2,
) -> np.ndarray:
    """Extract pooled convolution features for a batch of single-channel images."""
    if images.ndim != 3:
        raise ValueError("images must have shape (batch_size, height, width).")

    pooled_feature_maps = []
    for image in images:
        feature_maps = apply_filter_bank(
            image,
            kernels,
            biases=biases,
            stride=conv_stride,
            padding=conv_padding,
        )
        pooled_maps = [max_pool2d(relu(feature_map), kernel_size=pool_kernel_size, stride=pool_stride) for feature_map in feature_maps]
        pooled_feature_maps.append(np.stack(pooled_maps, axis=0))

    pooled_feature_maps = np.stack(pooled_feature_maps, axis=0)
    return flatten(pooled_feature_maps)


def linear_logits(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("bi,ij->bj", X, W) + b


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-12
    correct_class_probs = probs[np.arange(y_true.shape[0]), y_true]
    return float(-np.mean(np.log(correct_class_probs + eps)))


def linear_head_gradients(features: np.ndarray, probs: np.ndarray, y_true: np.ndarray):
    batch_size = features.shape[0]
    grad_logits = probs.copy()
    grad_logits[np.arange(batch_size), y_true] -= 1.0
    grad_logits /= batch_size
    grad_W = np.einsum("bi,bj->ij", features, grad_logits)
    grad_b = np.sum(grad_logits, axis=0)
    return grad_W, grad_b


def accuracy(logits: np.ndarray, y_true: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y_true))


def make_toy_image_dataset(
    samples_per_class: int = 40,
    image_size: int = 8,
    noise: float = 0.15,
    seed: int = 0,
):
    """Create three simple image classes: vertical, horizontal, and diagonal."""
    rng = np.random.default_rng(seed)
    images = []
    labels = []

    for _ in range(samples_per_class):
        image = noise * rng.normal(size=(image_size, image_size))
        image[:, image_size // 2] += 1.0
        images.append(image)
        labels.append(0)

    for _ in range(samples_per_class):
        image = noise * rng.normal(size=(image_size, image_size))
        image[image_size // 2, :] += 1.0
        images.append(image)
        labels.append(1)

    for _ in range(samples_per_class):
        image = noise * rng.normal(size=(image_size, image_size))
        for idx in range(image_size):
            image[idx, idx] += 1.0
        images.append(image)
        labels.append(2)

    return np.stack(images, axis=0), np.array(labels, dtype=int)


def run_demo(num_epochs: int = 100, lr: float = 0.5) -> None:
    images, labels = make_toy_image_dataset()

    kernels = np.array(
        [
            [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]],
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
            [[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]],
        ],
        dtype=float,
    )
    biases = np.zeros(kernels.shape[0], dtype=float)

    features = cnn_feature_extractor(images, kernels, biases=biases, conv_stride=1, conv_padding=1)
    num_features = features.shape[1]
    num_classes = int(labels.max()) + 1

    rng = np.random.default_rng(42)
    W = rng.normal(scale=0.01, size=(num_features, num_classes))
    b = np.zeros(num_classes, dtype=float)

    for epoch in range(num_epochs):
        logits = linear_logits(features, W, b)
        probs = softmax(logits)
        loss = cross_entropy(probs, labels)
        grad_W, grad_b = linear_head_gradients(features, probs, labels)
        W -= lr * grad_W
        b -= lr * grad_b

        if epoch == 0 or epoch == num_epochs - 1 or epoch % 20 == 0:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss:.4f} "
                f"acc={accuracy(logits, labels):.4f}"
            )

    single_image_feature_maps = apply_filter_bank(images[0], kernels, biases=biases, stride=1, padding=1)
    pooled_feature_maps = np.stack([max_pool2d(relu(feature_map)) for feature_map in single_image_feature_maps], axis=0)

    print("Feature shape after Conv+ReLU+Pool+Flatten:", features.shape)
    print("Single image conv feature maps shape:", single_image_feature_maps.shape)
    print("Single image pooled feature maps shape:", pooled_feature_maps.shape)


def describe_pipeline() -> None:
    x = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    pooled = max_pool2d(x, kernel_size=2, stride=2)
    print("Suggested pipeline: Conv -> ReLU -> Pool -> Flatten -> Linear")
    print("Example pooled result:\n", pooled)
    print("Now running a tiny CNN-style demo with fixed conv filters and a trainable linear head.\n")
    run_demo()


if __name__ == "__main__":
    describe_pipeline()
