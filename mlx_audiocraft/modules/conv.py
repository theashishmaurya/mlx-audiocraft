# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Convolution wrappers with causal/asymmetric padding.
#
# IMPORTANT: AudioCraft uses [B, C, T] convention throughout.
# MLX Conv1d uses [B, T, C]. We transpose at the boundary inside each
# wrapper so the rest of the codebase can keep using [B, C, T].

import math
import typing as tp
import warnings

import mlx.core as mx
import mlx.nn as nn

from ..utils.padding import get_extra_padding_for_conv1d, pad1d, unpad1d


# ── Normalization helpers ────────────────────────────────────────────────────

CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                 'time_group_norm'])


def get_norm_module(module: nn.Module, causal: bool = False,
                    norm: str = 'none', **norm_kwargs) -> nn.Module:
    """Return the proper normalization module.

    Note: ``weight_norm`` and ``spectral_norm`` are folded into the weights at
    load time (see ``utils/weight_convert.py``), so they are treated as ``none``
    here at runtime.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        # For Conv1d with out_channels: GroupNorm(1, C) = instance-norm per channel
        out_channels = _get_out_channels(module)
        return nn.GroupNorm(1, out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def _get_out_channels(module: nn.Module) -> int:
    """Extract out_channels from a Conv1d or ConvTranspose1d."""
    if hasattr(module, 'weight'):
        # MLX Conv1d weight shape: [out_channels, kernel_size, in_channels]
        return module.weight.shape[0]
    raise ValueError("Cannot determine out_channels from module")


# ── NormConv1d ───────────────────────────────────────────────────────────────

class NormConv1d(nn.Module):
    """Conv1d + normalization. Weight norm / spectral norm are folded at load time."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1,
                 bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
            dilation=dilation,
            bias=bias,
        )
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm
        # Store these for access by StreamableConv1d
        self._kernel_size = kernel_size
        self._stride = stride
        self._dilation = dilation

    def __call__(self, x: mx.array) -> mx.array:
        """x: [B, T, C] (MLX convention, already transposed by caller)."""
        x = self.conv(x)
        if self.norm_type == 'time_group_norm':
            # GroupNorm expects [B, T, C] which is what MLX uses
            x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """ConvTranspose1d + normalization."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm
        self._kernel_size = kernel_size
        self._stride = stride

    def __call__(self, x: mx.array) -> mx.array:
        """x: [B, T, C] (MLX convention)."""
        x = self.convtr(x)
        if self.norm_type == 'time_group_norm':
            x = self.norm(x)
        return x


# ── Streamable wrappers (with [B, C, T] <-> [B, T, C] transpose) ────────────

class StreamableConv1d(nn.Module):
    """Conv1d with built-in asymmetric/causal padding and normalization.

    Accepts and returns tensors in ``[B, C, T]`` (AudioCraft convention).
    Internally transposes to ``[B, T, C]`` for MLX Conv1d.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect'):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamableConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation}).")
        self.conv = NormConv1d(
            in_channels, out_channels, kernel_size, stride,
            dilation=dilation, groups=groups, bias=bias, causal=causal,
            norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: ``[B, C, T]`` input tensor.
        Returns:
            ``[B, C_out, T']`` output tensor.
        """
        B, C, T = x.shape
        kernel_size = self.conv._kernel_size
        stride = self.conv._stride
        dilation = self.conv._dilation
        effective_kernel = (kernel_size - 1) * dilation + 1
        padding_total = effective_kernel - stride
        extra_padding = get_extra_padding_for_conv1d(x, effective_kernel, stride, padding_total)

        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)

        # [B, C, T] -> [B, T, C] for MLX conv
        x = mx.transpose(x, axes=(0, 2, 1))
        x = self.conv(x)
        # [B, T', C_out] -> [B, C_out, T']
        x = mx.transpose(x, axes=(0, 2, 1))
        return x


class StreamableConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in asymmetric/causal padding trimming.

    Accepts and returns tensors in ``[B, C, T]``.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, causal: bool = False,
                 norm: str = 'none', trim_right_ratio: float = 1.,
                 norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels, out_channels, kernel_size, stride,
            causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert 0. <= self.trim_right_ratio <= 1.

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: ``[B, C, T]`` input tensor.
        Returns:
            ``[B, C_out, T']`` output tensor.
        """
        kernel_size = self.convtr._kernel_size
        stride = self.convtr._stride
        padding_total = kernel_size - stride

        # [B, C, T] -> [B, T, C]
        x = mx.transpose(x, axes=(0, 2, 1))
        y = self.convtr(x)
        # [B, T', C_out] -> [B, C_out, T']
        y = mx.transpose(y, axes=(0, 2, 1))

        # Trim padding
        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y
