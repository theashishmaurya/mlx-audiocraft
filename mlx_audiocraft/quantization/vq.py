# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Residual Vector Quantizer wrapper (inference only).

import math
import typing as tp

import mlx.core as mx
import mlx.nn as nn

from .core_vq import ResidualVectorQuantization


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer (inference only).

    Args:
        dimension: Codebook dimension.
        n_q: Number of residual quantizers.
        bins: Codebook size per quantizer.
    """

    def __init__(self, dimension: int = 256, n_q: int = 8,
                 q_dropout: bool = False, bins: int = 1024,
                 decay: float = 0.99, kmeans_init: bool = True,
                 kmeans_iters: int = 10,
                 threshold_ema_dead_code: float = 2.,
                 orthogonal_reg_weight: float = 0.0,
                 orthogonal_reg_active_codes_only: bool = False,
                 orthogonal_reg_max_codes: tp.Optional[int] = None):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins

        self.vq = ResidualVectorQuantization(
            dim=dimension,
            codebook_size=bins,
            num_quantizers=n_q,
            channels_last=False,
        )

    def encode(self, x: mx.array) -> mx.array:
        """Encode input to codebook indices.

        Args:
            x: ``[B, D, T]`` input.
        Returns:
            ``[B, K, T]`` codebook indices.
        """
        codes = self.vq.encode(x, n_q=self.n_q)  # [K, B, T]
        codes = mx.transpose(codes, axes=(1, 0, 2))  # [B, K, T]
        return codes

    def decode(self, codes: mx.array) -> mx.array:
        """Decode codebook indices to quantized representation.

        Args:
            codes: ``[B, K, T]`` codebook indices.
        Returns:
            ``[B, D, T]`` quantized output.
        """
        codes = mx.transpose(codes, axes=(1, 0, 2))  # [K, B, T]
        quantized = self.vq.decode(codes)
        return quantized

    @property
    def total_codebooks(self):
        return self.max_n_q

    @property
    def num_codebooks(self):
        return self.n_q

    def set_num_codebooks(self, n: int):
        assert 0 < n <= self.max_n_q
        self.n_q = n
