# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Vector Quantization (INFERENCE ONLY).
# Training logic (EMA updates, kmeans init, dead code expiry) is stripped.

import typing as tp

import mlx.core as mx
import mlx.nn as nn


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance (inference only).

    The codebook ``embed`` is loaded from converted weights.
    """

    def __init__(self, dim: int, codebook_size: int, **kwargs):
        super().__init__()
        self.codebook_size = codebook_size
        # Will be populated by weight loading
        self.embed = mx.zeros((codebook_size, dim))

    def encode(self, x: mx.array) -> mx.array:
        """Find nearest codebook entry for each vector.

        Args:
            x: ``[N, D]`` flattened input vectors.
        Returns:
            ``[N]`` codebook indices.
        """
        # Euclidean distance: ||x - e||^2 = ||x||^2 - 2*x.e + ||e||^2
        x_sq = mx.sum(x * x, axis=1, keepdims=True)  # [N, 1]
        e_sq = mx.sum(self.embed * self.embed, axis=1, keepdims=False)  # [C]
        dist = x_sq - 2 * (x @ self.embed.T) + e_sq[None, :]
        return mx.argmin(dist, axis=-1)

    def decode(self, embed_ind: mx.array) -> mx.array:
        """Look up codebook vectors by index.

        Args:
            embed_ind: Integer indices of any shape.
        Returns:
            Corresponding codebook vectors.
        """
        return self.embed[embed_ind]


class VectorQuantization(nn.Module):
    """Single-level vector quantization (inference only)."""

    def __init__(self, dim: int, codebook_size: int,
                 codebook_dim: tp.Optional[int] = None,
                 channels_last: bool = False, **kwargs):
        super().__init__()
        _codebook_dim = codebook_dim if codebook_dim is not None else dim
        requires_projection = _codebook_dim != dim

        self.project_in = nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        self.codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size)
        self.codebook_size = codebook_size
        self.channels_last = channels_last

    def encode(self, x: mx.array) -> mx.array:
        """Encode: ``[B, D, T]`` -> ``[B, T]`` indices."""
        if not self.channels_last:
            x = mx.transpose(x, axes=(0, 2, 1))  # [B, D, T] -> [B, T, D]
        x = self.project_in(x)
        # Flatten to [B*T, D] for codebook lookup
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        indices = self.codebook.encode(x_flat)
        return indices.reshape(B, T)

    def decode(self, embed_ind: mx.array) -> mx.array:
        """Decode: ``[B, T]`` indices -> ``[B, D, T]``."""
        quantize = self.codebook.decode(embed_ind)  # [B, T, D]
        quantize = self.project_out(quantize)
        if not self.channels_last:
            quantize = mx.transpose(quantize, axes=(0, 2, 1))  # [B, T, D] -> [B, D, T]
        return quantize


class ResidualVectorQuantization(nn.Module):
    """Multi-level residual VQ (inference only)."""

    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = [VectorQuantization(**kwargs) for _ in range(num_quantizers)]

    def encode(self, x: mx.array, n_q: tp.Optional[int] = None) -> mx.array:
        """Encode with residual quantization.

        Args:
            x: ``[B, D, T]`` input.
            n_q: Number of quantizers to use.
        Returns:
            ``[K, B, T]`` indices (K quantizer layers).
        """
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        return mx.stack(all_indices, axis=0)  # [K, B, T]

    def decode(self, q_indices: mx.array) -> mx.array:
        """Decode from stacked indices.

        Args:
            q_indices: ``[K, B, T]`` indices.
        Returns:
            ``[B, D, T]`` reconstructed output.
        """
        quantized_out = None
        for i in range(q_indices.shape[0]):
            indices = q_indices[i]  # [B, T]
            layer = self.layers[i]
            quantized = layer.decode(indices)
            if quantized_out is None:
                quantized_out = quantized
            else:
                quantized_out = quantized_out + quantized
        return quantized_out
