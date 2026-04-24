# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Streaming Transformer with causal attention and KV cache.

import typing as tp
import math

import mlx.core as mx
import mlx.nn as nn

from .rope import RotaryEmbedding
from .streaming import StreamingModule
from .activations import get_activation_fn


def create_norm_fn(norm_type: str, dim: int) -> nn.Module:
    if norm_type == 'layer_norm':
        return nn.LayerNorm(dim, eps=1e-5)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def create_sin_embedding(positions: mx.array, dim: int,
                         max_period: float = 10000) -> mx.array:
    """Create sinusoidal positional embedding with shape ``[B, T, C]``.

    Args:
        positions: ``[B, T, 1]`` position indices.
        dim: Embedding dimension (must be even).
        max_period: Maximum period for the frequencies.
    """
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.astype(mx.float32)
    adim = mx.arange(half_dim, dtype=mx.float32).reshape(1, 1, -1)
    phase = positions / (max_period ** (adim / (half_dim - 1)))
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


class LayerScale(nn.Module):
    """Learnable diagonal rescaling of residual outputs."""

    def __init__(self, channels: int, init: float = 1e-4, channel_last: bool = True):
        super().__init__()
        self.channel_last = channel_last
        self.scale = mx.full((channels,), init)

    def __call__(self, x: mx.array) -> mx.array:
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


def _expand_repeated_kv(x: mx.array, n_rep: int) -> mx.array:
    """Repeat KV heads to match query heads (GQA support).

    x: ``[B, T, n_kv_heads, head_dim]``
    Returns: ``[B, T, n_kv_heads * n_rep, head_dim]``
    """
    if n_rep == 1:
        return x
    B, T, n_kv_heads, head_dim = x.shape
    x = mx.expand_dims(x, axis=3)               # [B, T, n_kv, 1, D]
    x = mx.broadcast_to(x, (B, T, n_kv_heads, n_rep, head_dim))
    return x.reshape(B, T, n_kv_heads * n_rep, head_dim)


class StreamingMultiheadAttention(StreamingModule):
    """Multi-head attention with streaming KV cache, causal masking, and RoPE.

    All tensors use ``[B, T, H, D]`` layout for attention internals
    and ``[B, T, C]`` for input/output.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True, causal: bool = False,
                 past_context: tp.Optional[int] = None,
                 rope: tp.Optional[RotaryEmbedding] = None,
                 cross_attention: bool = False,
                 qk_layer_norm: bool = False, kv_repeat: int = 1,
                 # compat params (ignored in MLX)
                 custom: bool = False, memory_efficient: bool = False,
                 attention_as_float32: bool = False,
                 safe_streaming: bool = True, device=None, dtype=None):
        super().__init__()

        if past_context is not None:
            assert causal
        if cross_attention:
            assert not causal
            assert rope is None

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.past_context = past_context
        self.rope = rope
        self.cross_attention = cross_attention
        self.kv_repeat = kv_repeat
        self.head_dim = embed_dim // num_heads

        assert num_heads % kv_repeat == 0
        num_kv = num_heads // kv_repeat
        kv_dim = self.head_dim * num_kv

        # Combined Q/K/V projection
        out_dim = embed_dim + 2 * kv_dim
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.qk_layer_norm = qk_layer_norm
        if qk_layer_norm:
            self.q_layer_norm = nn.LayerNorm(embed_dim)
            self.k_layer_norm = nn.LayerNorm(embed_dim)

    def _get_mask(self, current_steps: int, total_steps: int) -> tp.Optional[mx.array]:
        """Build causal attention mask."""
        if not self.causal:
            return None
        if current_steps == 1 and self.past_context is None:
            # Single step, no mask needed (it's inherently causal)
            return None

        past_steps = total_steps - current_steps
        queries_pos = mx.arange(past_steps, total_steps).reshape(-1, 1)
        keys_pos = mx.arange(total_steps).reshape(1, -1)
        delta = queries_pos - keys_pos
        valid = delta >= 0
        if self.past_context is not None:
            valid = valid & (delta <= self.past_context)
        mask = mx.where(valid, mx.zeros((1,)), mx.full((1,), float('-inf')))
        return mask  # [current_steps, total_steps]

    def _complete_kv(self, k: mx.array, v: mx.array) -> tp.Tuple[mx.array, mx.array]:
        """Concatenate with cached past keys/values for streaming."""
        if self.cross_attention:
            return k, v

        time_dim = 1  # [B, T, H, D]
        if self._streaming_state:
            pk = self._streaming_state['past_keys']
            nk = mx.concatenate([pk, k], axis=time_dim)
            pv = self._streaming_state['past_values']
            nv = mx.concatenate([pv, v], axis=time_dim)
        else:
            nk = k
            nv = v

        offset = 0
        if self.past_context is not None:
            offset = max(0, nk.shape[time_dim] - self.past_context)

        if self._is_streaming:
            self._streaming_state['past_keys'] = nk[:, offset:]
            self._streaming_state['past_values'] = nv[:, offset:]
            if 'offset' in self._streaming_state:
                self._streaming_state['offset'] = self._streaming_state['offset'] + offset
            else:
                self._streaming_state['offset'] = mx.array(0)

        return nk, nv

    def __call__(self, query: mx.array, key: mx.array, value: mx.array,
                 need_weights: bool = False, attn_mask: tp.Optional[mx.array] = None):
        """
        Args:
            query, key, value: ``[B, T, C]`` tensors.
        Returns:
            Tuple of (output ``[B, T, C]``, None).
        """
        B, T, C = query.shape

        if self.cross_attention:
            # Separate projections for Q and K/V
            dim = self.embed_dim
            num_kv = self.num_heads // self.kv_repeat
            kv_dim = self.head_dim * num_kv

            # Split in_proj weight/bias into q, k, v parts
            proj = self.in_proj
            q = query @ proj.weight[:dim].T
            k = key @ proj.weight[dim:dim + kv_dim].T
            v = value @ proj.weight[dim + kv_dim:].T
            if hasattr(proj, 'bias'):
                q = q + proj.bias[:dim]
                k = k + proj.bias[dim:dim + kv_dim]
                v = v + proj.bias[dim + kv_dim:]

            if self.qk_layer_norm:
                q = self.q_layer_norm(q)
                k = self.k_layer_norm(k)

            q = q.reshape(B, T, self.num_heads, self.head_dim)
            k = k.reshape(B, -1, num_kv, self.head_dim)
            v = v.reshape(B, -1, num_kv, self.head_dim)
        else:
            # Self-attention: combined projection
            projected = self.in_proj(query)

            num_kv = self.num_heads // self.kv_repeat
            kv_dim = self.head_dim * num_kv

            q = projected[:, :, :self.embed_dim]
            k = projected[:, :, self.embed_dim:self.embed_dim + kv_dim]
            v = projected[:, :, self.embed_dim + kv_dim:]

            q = q.reshape(B, T, self.num_heads, self.head_dim)
            k = k.reshape(B, T, num_kv, self.head_dim)
            v = v.reshape(B, T, num_kv, self.head_dim)

            if self.qk_layer_norm:
                q_flat = q.reshape(B, T, -1)
                k_flat = k.reshape(B, T, -1)
                q_flat = self.q_layer_norm(q_flat)
                k_flat = self.k_layer_norm(k_flat)
                q = q_flat.reshape(B, T, self.num_heads, self.head_dim)
                k = k_flat.reshape(B, T, num_kv, self.head_dim)

            # Apply RoPE before KV cache concatenation
            if self.rope:
                past_keys_offset = 0
                past_context_offset = 0
                if 'past_keys' in self._streaming_state:
                    past_keys_offset = self._streaming_state['past_keys'].shape[1]
                if 'offset' in self._streaming_state:
                    past_context_offset = int(self._streaming_state['offset'].item())
                streaming_offset = past_context_offset + past_keys_offset
                q, k = self.rope.rotate_qk(q, k, start=streaming_offset, time_dim=1)

            # Complete KV with cache
            k, v = self._complete_kv(k, v)

            # Expand KV for GQA
            if self.kv_repeat > 1:
                k = _expand_repeated_kv(k, self.kv_repeat)
                v = _expand_repeated_kv(v, self.kv_repeat)

        # Build causal mask
        total_steps = k.shape[1]
        current_steps = q.shape[1]
        causal_mask = self._get_mask(current_steps, total_steps)

        # Combine with explicit attn_mask if provided
        if attn_mask is not None and causal_mask is not None:
            mask = causal_mask + attn_mask
        elif causal_mask is not None:
            mask = causal_mask
        else:
            mask = attn_mask

        # Scaled dot-product attention
        # q: [B, T_q, H, D], k: [B, T_k, H, D], v: [B, T_k, H, D]
        # Transpose to [B, H, T, D] for attention
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))

        scale = math.sqrt(self.head_dim)
        scores = (q @ mx.transpose(k, axes=(0, 1, 3, 2))) / scale  # [B, H, T_q, T_k]

        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        x = weights @ v  # [B, H, T_q, D]

        # Transpose back: [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        x = mx.transpose(x, axes=(0, 2, 1, 3))
        x = x.reshape(B, current_steps, self.embed_dim)

        x = self.out_proj(x)
        return x, None


class StreamingTransformerLayer(StreamingModule):
    """Single transformer layer with self-attention, optional cross-attention, and FFN."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, bias_ff: bool = True, bias_attn: bool = True,
                 causal: bool = False, past_context: tp.Optional[int] = None,
                 cross_attention: bool = False, layer_scale: tp.Optional[float] = None,
                 rope: tp.Optional[RotaryEmbedding] = None,
                 qk_layer_norm: bool = False, qk_layer_norm_cross: bool = False,
                 kv_repeat: int = 1, norm: str = 'layer_norm',
                 attention_dropout: tp.Optional[float] = None,
                 # compat params
                 custom: bool = False, memory_efficient: bool = False,
                 attention_as_float32: bool = False, device=None, dtype=None,
                 **kwargs):
        super().__init__()

        attn_dropout = dropout if attention_dropout is None else attention_dropout

        self.self_attn = StreamingMultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout,
            bias=bias_attn, causal=causal, past_context=past_context,
            rope=rope, qk_layer_norm=qk_layer_norm, kv_repeat=kv_repeat)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias_ff)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias_ff)

        self.norm1 = create_norm_fn(norm, d_model)
        self.norm2 = create_norm_fn(norm, d_model)

        self.layer_scale_1 = LayerScale(d_model, layer_scale) if layer_scale else nn.Identity()
        self.layer_scale_2 = LayerScale(d_model, layer_scale) if layer_scale else nn.Identity()

        self.norm_first = True  # AudioCraft always uses pre-norm

        # Cross attention
        self.cross_attention_module: tp.Optional[StreamingMultiheadAttention] = None
        self.norm_cross: tp.Optional[nn.Module] = None
        self.layer_scale_cross: tp.Optional[nn.Module] = None
        if cross_attention:
            self.cross_attention_module = StreamingMultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout,
                bias=bias_attn, cross_attention=True,
                qk_layer_norm=qk_layer_norm_cross)
            self.norm_cross = nn.LayerNorm(d_model, eps=1e-5)
            self.layer_scale_cross = (
                LayerScale(d_model, layer_scale) if layer_scale else nn.Identity()
            )

        # Activation for FFN
        self.activation = nn.GELU()

    def __call__(self, src: mx.array, src_mask: tp.Optional[mx.array] = None,
                 cross_attention_src: tp.Optional[mx.array] = None) -> mx.array:
        # Pre-norm self-attention
        x = src
        sa_input = self.norm1(x)
        sa_out, _ = self.self_attn(sa_input, sa_input, sa_input, attn_mask=src_mask)
        x = x + self.layer_scale_1(sa_out)

        # Cross-attention
        if self.cross_attention_module is not None and cross_attention_src is not None:
            ca_input = self.norm_cross(x)
            ca_out, _ = self.cross_attention_module(
                ca_input, cross_attention_src, cross_attention_src)
            x = x + self.layer_scale_cross(ca_out)

        # FFN
        ff_input = self.norm2(x)
        ff_out = self.linear2(self.activation(self.linear1(ff_input)))
        x = x + self.layer_scale_2(ff_out)

        return x


class StreamingTransformer(StreamingModule):
    """Transformer with streaming/causal support.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        dim_feedforward: FFN intermediate dimension.
        causal: Use causal attention.
        positional_embedding: 'sin', 'rope', or 'sin_rope'.
        cross_attention: Enable cross-attention in each layer.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 bias_ff: bool = True, bias_attn: bool = True,
                 causal: bool = False, past_context: tp.Optional[int] = None,
                 cross_attention: bool = False,
                 layer_scale: tp.Optional[float] = None,
                 positional_embedding: str = 'sin',
                 max_period: float = 10_000, positional_scale: float = 1.,
                 xpos: bool = False,
                 lr: tp.Optional[float] = None,
                 weight_decay: tp.Optional[float] = None,
                 norm: str = 'layer_norm',
                 # compat params
                 custom: bool = False, memory_efficient: bool = False,
                 attention_as_float32: bool = False,
                 layer_class: tp.Any = None, checkpointing: str = 'none',
                 device=None, dtype=None, **kwargs):
        super().__init__()
        assert d_model % num_heads == 0
        assert positional_embedding in ['sin', 'rope', 'sin_rope']

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale

        self.rope: tp.Optional[RotaryEmbedding] = None
        if positional_embedding in ['rope', 'sin_rope']:
            self.rope = RotaryEmbedding(
                d_model // num_heads, max_period=max_period,
                xpos=xpos, scale=positional_scale)

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(StreamingTransformerLayer(
                d_model=d_model, num_heads=num_heads,
                dim_feedforward=dim_feedforward, dropout=dropout,
                bias_ff=bias_ff, bias_attn=bias_attn,
                causal=causal, past_context=past_context,
                cross_attention=cross_attention,
                layer_scale=layer_scale, rope=self.rope,
                norm=norm, **kwargs))

    def __call__(self, x: mx.array, cross_attention_src: tp.Optional[mx.array] = None,
                 **kwargs) -> mx.array:
        B, T, C = x.shape

        if 'offsets' in self._streaming_state:
            offsets = self._streaming_state['offsets']
        else:
            offsets = mx.zeros((B,), dtype=mx.int32)

        if self.positional_embedding in ['sin', 'sin_rope']:
            positions = mx.arange(T).reshape(1, -1, 1)
            positions = positions + offsets.reshape(-1, 1, 1)
            pos_emb = create_sin_embedding(positions, C, max_period=self.max_period)
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, cross_attention_src=cross_attention_src)

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return x
