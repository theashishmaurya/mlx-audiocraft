# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – padding utilities for 1D convolutions.

import math
import typing as tp

import mlx.core as mx


def get_extra_padding_for_conv1d(x: mx.array, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """Compute extra padding needed so that the last convolution window is full.

    Args:
        x: Input tensor in **[B, C, T]** layout (AudioCraft convention).
        kernel_size: Effective kernel size.
        stride: Convolution stride.
        padding_total: Total padding already applied.
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: mx.array, kernel_size: int, stride: int,
                   padding_total: int = 0) -> mx.array:
    """Pad *x* so the last convolution window is full (extra padding on the right)."""
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    if extra_padding > 0:
        x = pad1d(x, (0, extra_padding), mode='constant', value=0.)
    return x


# ── 1-D pad / unpad ─────────────────────────────────────────────────────────

def _reflect_pad_1d(x: mx.array, pad_left: int, pad_right: int) -> mx.array:
    """Reflect-pad the last dimension of *x*.

    Handles the edge case where ``pad > length - 1`` by first zero-padding the
    input so that reflect padding becomes valid (matching the PyTorch behaviour
    used by the original AudioCraft).

    x layout: ``[..., T]``
    """
    length = x.shape[-1]
    max_pad = max(pad_left, pad_right)
    extra = 0
    if length <= max_pad:
        extra = max_pad - length + 1
        # Zero-pad on the right so we have enough elements to reflect
        zeros_shape = list(x.shape)
        zeros_shape[-1] = extra
        x = mx.concatenate([x, mx.zeros(zeros_shape, dtype=x.dtype)], axis=-1)

    parts = []
    if pad_left > 0:
        # Reflect from index 1..pad_left (exclusive of index 0), reversed
        left = x[..., pad_left: 0: -1]
        parts.append(left)
    parts.append(x)
    if pad_right > 0:
        # Reflect from end, reversed
        right = x[..., -2: -(pad_right + 2): -1]
        parts.append(right)

    padded = mx.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]

    # Remove the temporary extra zero-padding we introduced
    if extra > 0:
        end = padded.shape[-1] - extra
        padded = padded[..., :end]
    return padded


def pad1d(x: mx.array, paddings: tp.Tuple[int, int],
          mode: str = 'constant', value: float = 0.) -> mx.array:
    """Pad the last dimension of *x*.

    Supports ``mode='constant'`` and ``mode='reflect'``.
    *x* can have any number of leading dimensions; padding is applied to the
    **last** dimension only (matching the ``[B, C, T]`` AudioCraft convention).
    """
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)

    if padding_left == 0 and padding_right == 0:
        return x

    if mode == 'reflect':
        return _reflect_pad_1d(x, padding_left, padding_right)

    # Constant padding
    ndim = x.ndim
    pad_widths = [(0, 0)] * (ndim - 1) + [(padding_left, padding_right)]
    return mx.pad(x, pad_widths, constant_values=value)


def unpad1d(x: mx.array, paddings: tp.Tuple[int, int]) -> mx.array:
    """Remove padding from the last dimension of *x*."""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]
