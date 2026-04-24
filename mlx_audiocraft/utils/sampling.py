# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – sampling utilities for token generation.

import mlx.core as mx


def sample_top_k(logits: mx.array, k: int) -> mx.array:
    """Zero-out all logits except the top-*k* values (per row).

    Args:
        logits: ``[B, vocab]`` or ``[..., vocab]`` logits.
        k: Number of top values to keep.
    Returns:
        Filtered logits with ``-inf`` in masked positions.
    """
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    # topk returns (values, indices)
    top_values = mx.topk(logits, k=k, axis=-1)
    # Get the k-th largest value as threshold
    kth = mx.min(top_values, axis=-1, keepdims=True)
    return mx.where(logits >= kth, logits, mx.array(float('-inf'), dtype=logits.dtype))


def sample_top_p(logits: mx.array, p: float) -> mx.array:
    """Nucleus sampling: keep the smallest set of tokens with cumulative
    probability >= *p*.

    Args:
        logits: ``[B, vocab]`` logits (un-normalised).
        p: Probability threshold in ``(0, 1]``.
    Returns:
        Filtered logits with ``-inf`` in masked positions.
    """
    if p >= 1.0:
        return logits

    # Sort descending
    sorted_indices = mx.argsort(logits, axis=-1)
    # argsort is ascending, reverse for descending
    sorted_indices = sorted_indices[..., ::-1]
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)

    # Cumulative softmax probabilities
    probs = mx.softmax(sorted_logits, axis=-1)
    cumulative_probs = mx.cumsum(probs, axis=-1)

    # Mask tokens with cumulative probability above threshold
    # Keep at least the first token
    sorted_mask = cumulative_probs - probs >= p
    sorted_logits = mx.where(sorted_mask, mx.array(float('-inf'), dtype=logits.dtype), sorted_logits)

    # Un-sort back to original order
    # Create inverse permutation
    inv_indices = mx.argsort(sorted_indices, axis=-1)
    return mx.take_along_axis(sorted_logits, inv_indices, axis=-1)


def multinomial(logits: mx.array, num_samples: int = 1,
                temperature: float = 1.0) -> mx.array:
    """Sample from a categorical distribution defined by *logits*.

    Args:
        logits: ``[B, vocab]`` un-normalised log-probabilities.
        num_samples: Number of samples to draw per batch element.
        temperature: Sampling temperature (applied before softmax).
    Returns:
        ``[B, num_samples]`` sampled token indices.
    """
    if temperature != 1.0:
        logits = logits / temperature

    # mx.random.categorical expects log-probabilities along the last axis
    # and returns a single sample per call
    if num_samples == 1:
        return mx.random.categorical(logits)[:, None]

    samples = []
    for _ in range(num_samples):
        s = mx.random.categorical(logits)
        samples.append(s)
    return mx.stack(samples, axis=-1)
