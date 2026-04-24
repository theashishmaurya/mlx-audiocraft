# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Transformer Language Model for music generation (inference only).
# Training logic (init_weights, FSDP, compute_predictions) is stripped.

import logging
import math
import typing as tp

import mlx.core as mx
import mlx.nn as nn

from ..modules.streaming import StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..modules.codebooks_patterns import CodebooksPatternProvider
from ..modules.activations import get_activation_fn
from ..utils.sampling import sample_top_k, sample_top_p, multinomial

logger = logging.getLogger(__name__)

# Condition type: (condition_tensor, mask) both mx.array
ConditionType = tp.Tuple[mx.array, mx.array]
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


class LMModel(StreamingModule):
    """Transformer-based language model on multiple streams of codes (inference only).

    Args:
        pattern_provider: Pattern provider for codebook interleaving.
        condition_provider: Conditioning provider from metadata.
        fuser: Fuser handling the fusing of conditions with language model input.
        n_q: Number of parallel streams to model.
        card: Cardinality, vocabulary size.
        dim: Dimension of the transformer encoder.
        num_heads: Number of heads for the transformer encoder.
        hidden_scale: Scale for hidden feed forward dimension.
        norm: Normalization method.
        norm_first: Use pre-norm instead of post-norm.
        bias_proj: Use bias for output projections.
        cfg_coef: Classifier-free guidance coefficient.
        two_step_cfg: Whether to run CFG with 2 distinct steps.
        **kwargs: Additional parameters for the transformer encoder.
    """

    def __init__(self, pattern_provider: CodebooksPatternProvider,
                 condition_provider,  # ConditioningProvider
                 fuser,               # ConditionFuser
                 n_q: int = 8, card: int = 1024, dim: int = 128,
                 num_heads: int = 8, hidden_scale: int = 4,
                 norm: str = 'layer_norm', norm_first: bool = False,
                 emb_lr: tp.Optional[float] = None,
                 bias_proj: bool = True,
                 weight_init: tp.Optional[str] = None,
                 depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False,
                 cfg_dropout: float = 0, cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {},
                 two_step_cfg: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg_coef = cfg_coef
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.card = card
        embed_dim = self.card + 1  # +1 for special token
        self.n_q = n_q
        self.dim = dim
        self.pattern_provider = pattern_provider
        self.two_step_cfg = two_step_cfg

        # Embeddings: one per codebook
        self.emb = [nn.Embedding(embed_dim, dim) for _ in range(n_q)]

        # Activation fn mapping for transformer
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])

        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm, norm_first=norm_first, **kwargs)

        # Output normalization (only when using pre-norm)
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)

        # Output projections: one per codebook
        self.linears = [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)]

    @property
    def special_token_id(self) -> int:
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def __call__(self, sequence: mx.array,
                 conditions: tp.List = [],
                 condition_tensors: tp.Optional[ConditionTensors] = None,
                 stage: int = -1) -> mx.array:
        """Apply language model on sequence and conditions.

        Args:
            sequence: ``[B, K, S]`` codebook indices.
            conditions: List of conditioning attributes (ignored if condition_tensors given).
            condition_tensors: Pre-computed conditioning tensors.
            stage: Codebook level (ignored, kept for API compat).
        Returns:
            ``[B, K, S, card]`` logits.
        """
        B, K, S = sequence.shape
        assert K == self.num_codebooks

        # Sum embeddings from all codebooks -> [B, S, D]
        input_ = self.emb[0](sequence[:, 0])
        for k in range(1, K):
            input_ = input_ + self.emb[k](sequence[:, k])

        if condition_tensors is None:
            assert not self._is_streaming, \
                "Condition tensors should be precomputed when streaming."
            # Tokenize and encode conditions
            tokenized = self.condition_provider.tokenize(conditions)
            condition_tensors = self.condition_provider(tokenized)
        else:
            assert not conditions, \
                "Shouldn't pass both conditions and condition_tensors."

        # Fuse conditions with input
        input_, cross_attention_input = self.fuser(input_, condition_tensors)

        # Transformer forward
        out = self.transformer(
            input_, cross_attention_src=cross_attention_input)

        # Output norm (pre-norm mode)
        if self.out_norm:
            out = self.out_norm(out)

        # Project to logits: [B, K, S, card]
        logits = mx.stack(
            [self.linears[k](out) for k in range(K)], axis=1)

        # Remove prefix from model outputs (prepended conditions)
        if len(self.fuser.fuse2cond.get('prepend', [])) > 0:
            logits = logits[:, :, -S:]

        return logits

    def _sample_next_token(self,
                           sequence: mx.array,
                           cfg_conditions: CFGConditions,
                           unconditional_state: State,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: tp.Optional[float] = None,
                           cfg_coef_beta: tp.Optional[float] = None,
                           two_step_cfg: tp.Optional[bool] = None) -> mx.array:
        """Sample next token from the model.

        Supports classifier-free guidance (single-pass, two-step, and style CFG).

        Args:
            sequence: ``[B, K, S]`` current sequence (S=1 when streaming).
            cfg_conditions: Pre-computed condition tensors for CFG.
            unconditional_state: Streaming state for unconditional pass (two-step CFG).
            use_sampling: Whether to sample or use greedy decoding.
            temp: Sampling temperature.
            top_k: K for top-k sampling.
            top_p: P for top-p (nucleus) sampling.
            cfg_coef: Classifier-free guidance coefficient.
            cfg_coef_beta: Style CFG beta coefficient (double CFG).
            two_step_cfg: Whether to use two-step CFG.
        Returns:
            ``[B, K, 1]`` next token indices.
        """
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg

        if cfg_coef_beta is not None:
            # Style CFG: triple batch (cond_text+style, cond_style_only, uncond)
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                sequence = mx.concatenate(
                    [sequence, sequence, sequence], axis=0)
            all_logits = self(
                sequence, conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits = all_logits[:B]
                wav_logits = all_logits[B:2*B]
                uncond_logits = all_logits[2*B:]
                logits = uncond_logits + cfg_coef * (
                    wav_logits + cfg_coef_beta * (cond_logits - wav_logits)
                    - uncond_logits)
            else:
                logits = all_logits

        elif two_step_cfg and cfg_conditions != {}:
            # Two-step CFG: separate forward passes for cond and uncond
            assert isinstance(cfg_conditions, tuple), type(cfg_conditions)
            condition_tensors, null_condition_tensors = cfg_conditions
            cond_logits = self(
                sequence, conditions=[], condition_tensors=condition_tensors)
            state = self.get_streaming_state()
            self.set_streaming_state(unconditional_state)
            uncond_logits = self(
                sequence, conditions=[],
                condition_tensors=null_condition_tensors)
            unconditional_state.update(self.get_streaming_state())
            self.set_streaming_state(state)
            logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_coef

        else:
            # Default single-pass CFG: double batch (cond, uncond)
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                sequence = mx.concatenate(
                    [sequence, sequence], axis=0)
            all_logits = self(
                sequence, conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits = all_logits[:B]
                uncond_logits = all_logits[B:]
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
            else:
                logits = all_logits

        # logits: [B, K, S, card] -> take last step -> [B, K, card]
        logits = logits[:, :, -1, :]

        # Sample or greedy decode
        if use_sampling and temp > 0.0:
            # Apply temperature
            logits = logits / temp
            # Sample each codebook independently
            next_token_list = []
            for k in range(logits.shape[1]):
                k_logits = logits[:, k, :]  # [B, card]
                if top_p > 0.0:
                    k_logits = sample_top_p(k_logits, top_p)
                elif top_k > 0:
                    k_logits = sample_top_k(k_logits, top_k)
                # Sample from filtered distribution
                token = mx.random.categorical(k_logits)  # [B]
                next_token_list.append(token)
            next_token = mx.stack(next_token_list, axis=1)  # [B, K]
            next_token = next_token[:, :, None]  # [B, K, 1]
        else:
            next_token = mx.argmax(logits, axis=-1)  # [B, K]
            next_token = next_token[:, :, None]  # [B, K, 1]

        return next_token

    def generate(self,
                 prompt: tp.Optional[mx.array] = None,
                 conditions: tp.List = [],
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 use_sampling: bool = True,
                 temp: float = 1.0,
                 top_k: int = 250,
                 top_p: float = 0.0,
                 cfg_coef: tp.Optional[float] = None,
                 cfg_coef_beta: tp.Optional[float] = None,
                 two_step_cfg: tp.Optional[bool] = None,
                 remove_prompts: bool = False,
                 check: bool = False,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                 ) -> mx.array:
        """Generate tokens autoregressively.

        Args:
            prompt: ``[B, K, T]`` prompt tokens (optional).
            conditions: List of conditioning attributes.
            num_samples: Number of samples (used if no prompt/conditions).
            max_gen_len: Maximum generation length in timesteps.
            use_sampling: Whether to use sampling or greedy decoding.
            temp: Sampling temperature.
            top_k: K for top-k sampling.
            top_p: P for top-p (nucleus) sampling.
            cfg_coef: Classifier-free guidance coefficient.
            cfg_coef_beta: Style CFG beta coefficient.
            two_step_cfg: Whether to use two-step CFG.
            remove_prompts: Whether to strip prompts from output.
            check: Whether to run sanity checks.
            callback: Progress callback ``(current_step, total_steps)``.
        Returns:
            ``[B, K, T]`` generated codebook tokens.
        """
        # Determine batch size
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert all(x == possible_num_samples[0] for x in possible_num_samples)
        num_samples = possible_num_samples[0]

        # ── Prepare CFG conditions ──────────────────────────────────────
        # Lazy import to avoid circular dependency
        from ..modules.conditioners import (
            ClassifierFreeGuidanceDropout, _drop_description_condition
        )

        cfg_conditions: CFGConditions = {}
        if cfg_coef_beta is not None:
            # Style CFG: triple (cond_text+style, cond_style_only, uncond)
            if conditions:
                wav_conditions = _drop_description_condition(conditions)
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                conditions_3x = conditions + wav_conditions + null_conditions
                tokenized = self.condition_provider.tokenize(conditions_3x)
                cfg_conditions = self.condition_provider(tokenized)
        elif conditions:
            two_step = self.two_step_cfg if two_step_cfg is None else two_step_cfg
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            if two_step:
                cfg_conditions = (
                    self.condition_provider(
                        self.condition_provider.tokenize(conditions)),
                    self.condition_provider(
                        self.condition_provider.tokenize(null_conditions)),
                )
            else:
                conditions_2x = conditions + null_conditions
                tokenized = self.condition_provider.tokenize(conditions_2x)
                cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        # ── Initialize prompt ───────────────────────────────────────────
        if prompt is None:
            assert num_samples > 0
            prompt = mx.zeros((num_samples, self.num_codebooks, 0),
                              dtype=mx.int32)

        B, K, T = prompt.shape
        start_offset = T
        assert start_offset < max_gen_len

        # ── Build pattern sequence ──────────────────────────────────────
        pattern = self.pattern_provider.get_pattern(max_gen_len)
        unknown_token = -1

        # gen_codes: [B, K, max_gen_len] filled with unknown
        gen_codes = mx.full((B, K, max_gen_len), unknown_token, dtype=mx.int32)
        # Fill in the prompt
        if T > 0:
            gen_codes = _set_slice(gen_codes, prompt, t_end=T)

        # Build interleaved pattern sequence: [B, K, S]
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(
            gen_codes, self.special_token_id)

        # Find where in the sequence the prompt ends
        start_offset_sequence = pattern.get_first_step_with_timesteps(
            start_offset)
        assert start_offset_sequence is not None

        # ── Autoregressive generation loop ──────────────────────────────
        with self.streaming():
            unconditional_state = self.get_streaming_state()
            prev_offset = 0
            gen_sequence_len = gen_sequence.shape[-1]

            for offset in range(start_offset_sequence, gen_sequence_len):
                # Current slice for this step
                curr_sequence = gen_sequence[:, :, prev_offset:offset]
                curr_mask = mx.broadcast_to(
                    mask[None, :, prev_offset:offset],
                    (B, K, offset - prev_offset))

                if check:
                    expected = mx.where(curr_mask, curr_sequence,
                                        mx.array(self.special_token_id))
                    assert mx.array_equal(curr_sequence, expected)
                    assert not mx.any(curr_sequence == unknown_token)

                # Sample next token: [B, K, 1]
                next_token = self._sample_next_token(
                    curr_sequence, cfg_conditions, unconditional_state,
                    use_sampling, temp, top_k, top_p,
                    cfg_coef=cfg_coef, cfg_coef_beta=cfg_coef_beta,
                    two_step_cfg=two_step_cfg)

                # Force evaluation for the autoregressive dependency
                mx.eval(next_token)

                # Mask invalid positions with special_token_id
                valid_mask = mx.broadcast_to(
                    mask[None, :, offset:offset+1], (B, K, 1))
                next_token = mx.where(
                    valid_mask,
                    next_token,
                    mx.array(self.special_token_id, dtype=next_token.dtype))

                # Only write over unknown tokens (don't overwrite prompt)
                current_val = gen_sequence[:, :, offset:offset+1]
                next_token = mx.where(
                    current_val == unknown_token,
                    next_token, current_val)

                gen_sequence = _set_slice_at(
                    gen_sequence, next_token, offset)

                prev_offset = offset
                if callback is not None:
                    callback(
                        1 + offset - start_offset_sequence,
                        gen_sequence_len - start_offset_sequence)

        # ── Revert pattern to original code layout ──────────────────────
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(
            gen_sequence, special_token=unknown_token)

        if check:
            assert mx.all(out_codes[:, :, :max_gen_len] != unknown_token)
            assert mx.all(out_mask[:, :max_gen_len] == 1)

        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[:, :, out_start_offset:max_gen_len]

        if check:
            assert mx.all(out_codes >= 0)
            assert mx.all(out_codes <= self.card)

        return out_codes


# ── Helper functions for in-place-like updates on immutable arrays ──────────

def _set_slice(arr: mx.array, values: mx.array, t_end: int) -> mx.array:
    """Set arr[:, :, :t_end] = values. Returns new array (MLX is functional)."""
    if t_end >= arr.shape[2]:
        return values
    rest = arr[:, :, t_end:]
    return mx.concatenate([values, rest], axis=2)


def _set_slice_at(arr: mx.array, value: mx.array, t: int) -> mx.array:
    """Set arr[:, :, t:t+1] = value. Returns new array."""
    parts = []
    if t > 0:
        parts.append(arr[:, :, :t])
    parts.append(value)
    if t + 1 < arr.shape[2]:
        parts.append(arr[:, :, t+1:])
    return mx.concatenate(parts, axis=2)
