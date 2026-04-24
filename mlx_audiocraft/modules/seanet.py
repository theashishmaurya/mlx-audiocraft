# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – SEANet encoder/decoder for EnCodec.
# All tensors use [B, C, T] layout (AudioCraft convention).

import typing as tp

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .conv import StreamableConv1d, StreamableConvTranspose1d
from .lstm import StreamableLSTM


def _get_activation(name: str, params: dict) -> nn.Module:
    """Get activation by name. MLX doesn't have all PyTorch activations
    with arbitrary kwargs, so we map the common ones."""
    name_lower = name.lower()
    if name_lower == 'elu':
        return nn.ELU()
    elif name_lower == 'relu':
        return nn.ReLU()
    elif name_lower == 'gelu':
        return nn.GELU()
    elif name_lower == 'silu' or name_lower == 'swish':
        return nn.SiLU()
    elif name_lower == 'tanh':
        return nn.Tanh()
    elif name_lower == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.

    Args:
        dim: Input/output dimension.
        kernel_sizes: Kernel sizes for the convolutions.
        dilations: Dilation for each convolution.
        activation: Activation function name.
        activation_params: Params for activation (used in PyTorch, mapped here).
        norm: Normalization method.
        norm_params: Normalization parameters.
        causal: Whether to use causal convolutions.
        pad_mode: Padding mode.
        compress: Compression ratio for hidden dim.
        true_skip: Use identity shortcut (True) or 1x1 conv (False).
    """

    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1],
                 dilations: tp.List[int] = [1, 1],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {},
                 causal: bool = False, pad_mode: str = 'reflect',
                 compress: int = 2, true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        hidden = dim // compress
        block_layers = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block_layers.append(_get_activation(activation, activation_params))
            block_layers.append(StreamableConv1d(
                in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode))
        self.block = block_layers

        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamableConv1d(
                dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

    def __call__(self, x: mx.array) -> mx.array:
        y = x
        for layer in self.block:
            y = layer(y)
        return self.shortcut(x) + y


class SEANetEncoder(nn.Module):
    """SEANet encoder: audio ``[B, C, T]`` -> latent ``[B, D, T']``."""

    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32,
                 n_residual_layers: int = 3, ratios: tp.List[int] = [8, 5, 4, 2],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {},
                 kernel_size: int = 7, last_kernel_size: int = 7,
                 residual_kernel_size: int = 3, dilation_base: int = 2,
                 causal: bool = False, pad_mode: str = 'reflect',
                 true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2
        self.disable_norm_outer_blocks = disable_norm_outer_blocks

        mult = 1
        model: tp.List[nn.Module] = [
            StreamableConv1d(channels, mult * n_filters, kernel_size,
                             norm='none' if disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if disable_norm_outer_blocks >= i + 2 else norm
            for j in range(n_residual_layers):
                model.append(SEANetResnetBlock(
                    mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                    dilations=[dilation_base ** j, 1],
                    norm=block_norm, norm_params=norm_params,
                    activation=activation, activation_params=activation_params,
                    causal=causal, pad_mode=pad_mode,
                    compress=compress, true_skip=true_skip))
            model.append(_get_activation(activation, activation_params))
            model.append(StreamableConv1d(
                mult * n_filters, mult * n_filters * 2,
                kernel_size=ratio * 2, stride=ratio,
                norm=block_norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode))
            mult *= 2

        if lstm:
            model.append(StreamableLSTM(mult * n_filters, num_layers=lstm))

        model.append(_get_activation(activation, activation_params))
        model.append(StreamableConv1d(
            mult * n_filters, dimension, last_kernel_size,
            norm='none' if disable_norm_outer_blocks == self.n_blocks else norm,
            norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode))

        self.model = model

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.model:
            x = layer(x)
        return x


class SEANetDecoder(nn.Module):
    """SEANet decoder: latent ``[B, D, T']`` -> audio ``[B, C, T]``."""

    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32,
                 n_residual_layers: int = 3, ratios: tp.List[int] = [8, 5, 4, 2],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None,
                 final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {},
                 kernel_size: int = 7, last_kernel_size: int = 7,
                 residual_kernel_size: int = 3, dilation_base: int = 2,
                 causal: bool = False, pad_mode: str = 'reflect',
                 true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0,
                 trim_right_ratio: float = 1.0):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2
        self.disable_norm_outer_blocks = disable_norm_outer_blocks

        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamableConv1d(dimension, mult * n_filters, kernel_size,
                             norm='none' if disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        if lstm:
            model.append(StreamableLSTM(mult * n_filters, num_layers=lstm))

        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            model.append(_get_activation(activation, activation_params))
            model.append(StreamableConvTranspose1d(
                mult * n_filters, mult * n_filters // 2,
                kernel_size=ratio * 2, stride=ratio,
                norm=block_norm, norm_kwargs=norm_params,
                causal=causal, trim_right_ratio=trim_right_ratio))
            for j in range(n_residual_layers):
                model.append(SEANetResnetBlock(
                    mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                    dilations=[dilation_base ** j, 1],
                    activation=activation, activation_params=activation_params,
                    norm=block_norm, norm_params=norm_params,
                    causal=causal, pad_mode=pad_mode,
                    compress=compress, true_skip=true_skip))
            mult //= 2

        model.append(_get_activation(activation, activation_params))
        model.append(StreamableConv1d(
            n_filters, channels, last_kernel_size,
            norm='none' if disable_norm_outer_blocks >= 1 else norm,
            norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode))

        if final_activation is not None:
            model.append(_get_activation(final_activation, final_activation_params or {}))

        self.model = model

    def __call__(self, z: mx.array) -> mx.array:
        for layer in self.model:
            z = layer(z)
        return z
