# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Rotary Positional Embeddings (RoPE) without complex tensors.

import typing as tp

import mlx.core as mx
import mlx.nn as nn


class XPos(nn.Module):
    """Length-extrapolatable positional embedding (xPos).

    Applies an exponential decay to the RoPE rotation matrix.
    """

    def __init__(self, dim: int, smoothing: float = 0.4, base_scale: int = 512):
        super().__init__()
        assert dim % 2 == 0
        self.base_scale = base_scale

        half_dim = dim // 2
        adim = mx.arange(half_dim, dtype=mx.float32)
        self.decay_rates = (adim / half_dim + smoothing) / (1.0 + smoothing)
        self._decay_cache: tp.Optional[mx.array] = None
        self._decay_cache_len: int = 0

    def get_decay(self, start: int, end: int) -> tp.Tuple[mx.array, mx.array]:
        """Return (decay_cos, decay_sin) each of shape ``[T, half_dim]``.

        Since we decomposed complex numbers into cos/sin pairs, decay is purely
        real (zero angle), so decay_sin = 0.
        """
        if self._decay_cache is None or end > self._decay_cache_len:
            idx = mx.arange(end, dtype=mx.float32)
            power = idx / self.base_scale
            # scale shape: [end, half_dim]
            scale = self.decay_rates[None, :] ** power[:, None]
            self._decay_cache = scale
            self._decay_cache_len = end
        return self._decay_cache[start:end]


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) – MLX version using sin/cos pairs.

    Instead of complex arithmetic (``torch.view_as_complex`` / ``torch.polar``),
    we decompose the rotation into explicit sin/cos operations:

        x_rotated[..., 2i]   = x[..., 2i] * cos - x[..., 2i+1] * sin
        x_rotated[..., 2i+1] = x[..., 2i] * sin + x[..., 2i+1] * cos

    Args:
        dim: Embedding dimension (must be even).
        max_period: Maximum period of the rotation frequencies.
        xpos: Use xPos exponential decay.
        scale: Scale of positional embedding (0 to deactivate).
    """

    def __init__(self, dim: int, max_period: float = 10000.0,
                 xpos: bool = False, scale: float = 1.0):
        super().__init__()
        assert dim % 2 == 0
        self.scale = scale
        self.dim = dim

        half_dim = dim // 2
        adim = mx.arange(0, dim, 2, dtype=mx.float32)[:half_dim]
        self.frequencies = 1.0 / (max_period ** (adim / dim))

        self._cos_cache: tp.Optional[mx.array] = None
        self._sin_cache: tp.Optional[mx.array] = None
        self._cache_len: int = 0

        self.xpos = XPos(dim) if xpos else None

    def _build_cache(self, end: int):
        """Pre-compute cos/sin tables up to *end* positions."""
        if self._cos_cache is not None and end <= self._cache_len:
            return
        idx = mx.arange(end, dtype=mx.float32)
        # angles: [end, half_dim]
        angles = idx[:, None] * self.frequencies[None, :]
        self._cos_cache = mx.cos(angles)
        self._sin_cache = mx.sin(angles)
        self._cache_len = end

    def rotate(self, x: mx.array, start: int = 0, time_dim: int = 1,
               invert_decay: bool = False) -> mx.array:
        """Apply RoPE rotation to *x*.

        Args:
            x: ``[B, T, H, D]`` or ``[B, T, D]`` tensor.
            start: Starting position index (for streaming).
            time_dim: Which dimension is the time/sequence dimension.
            invert_decay: Invert xPos decay (used for keys).
        """
        T = x.shape[time_dim]
        end = start + T
        self._build_cache(end)

        cos = self._cos_cache[start:end]  # [T, half_dim]
        sin = self._sin_cache[start:end]  # [T, half_dim]

        # Reshape cos/sin to broadcast with x
        # x is typically [B, T, H, D] or [B, T, D]
        target_shape = [1] * x.ndim
        target_shape[time_dim] = T
        target_shape[-1] = -1
        cos = cos.reshape(target_shape)  # e.g. [1, T, 1, half_dim]
        sin = sin.reshape(target_shape)

        # Apply xPos decay if enabled
        if self.xpos is not None:
            decay = self.xpos.get_decay(start, end)
            decay = decay.reshape(target_shape)
            if invert_decay:
                decay = 1.0 / decay
            cos = cos * decay
            sin = sin * decay

        # Apply scale
        if self.scale != 1.0:
            cos = cos * self.scale + (1.0 - self.scale)
            sin = sin * self.scale

        # Split x into even/odd pairs
        x1 = x[..., 0::2]  # even indices
        x2 = x[..., 1::2]  # odd indices

        # Apply rotation: (x1 + i*x2) * (cos + i*sin)
        # Real part: x1*cos - x2*sin
        # Imag part: x1*sin + x2*cos
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Interleave back: [x1_0, x2_0, x1_1, x2_1, ...]
        out = mx.zeros_like(x)
        out = out.at[..., 0::2].add(out1)
        out = out.at[..., 1::2].add(out2)

        return out.astype(x.dtype)

    def rotate_qk(self, query: mx.array, key: mx.array,
                   start: int = 0, time_dim: int = 1) -> tp.Tuple[mx.array, mx.array]:
        """Apply RoPE to both query and key tensors.

        Supports streaming mode where key may be longer than query due to
        cached past timesteps.

        Args:
            query: Query tensor.
            key: Key tensor (may include past cached steps).
            start: Start index for time offset.
            time_dim: Which dimension is time.
        """
        query_timesteps = query.shape[time_dim]
        key_timesteps = key.shape[time_dim]
        streaming_offset = key_timesteps - query_timesteps

        query_out = self.rotate(query, start + streaming_offset, time_dim)
        key_out = self.rotate(key, start, time_dim, invert_decay=True)

        return query_out, key_out
