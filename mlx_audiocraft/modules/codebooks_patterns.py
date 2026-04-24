# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Codebook interleaving patterns.
# This is mostly pure Python logic; tensor ops use mlx.core instead of torch.

from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
import logging
import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import mlx.core as mx

LayoutCoord = namedtuple('LayoutCoord', ['t', 'q'])  # (timestep, codebook index)
PatternLayout = tp.List[tp.List[LayoutCoord]]
logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Base implementation of a pattern over a sequence with multiple codebooks.

    The codebook pattern consists in a layout, defining for each sequence step
    the list of coordinates of each codebook timestep in the resulting interleaved
    sequence. The first item of the pattern is always an empty list in order to
    properly insert a special token to start with.
    """
    layout: PatternLayout
    timesteps: int
    n_q: int

    def __post_init__(self):
        assert len(self.layout) > 0
        self._validate_layout()
        self._build_reverted_sequence_scatter_indexes = \
            lru_cache(100)(self._build_reverted_sequence_scatter_indexes)
        self._build_pattern_sequence_scatter_indexes = \
            lru_cache(100)(self._build_pattern_sequence_scatter_indexes)
        logger.info("New pattern, time steps: %d, sequence steps: %d",
                    self.timesteps, len(self.layout))

    def _validate_layout(self):
        q_timesteps = {q: 0 for q in range(self.n_q)}
        for s, seq_coords in enumerate(self.layout):
            if len(seq_coords) > 0:
                qs = set()
                for coord in seq_coords:
                    qs.add(coord.q)
                    last_q_timestep = q_timesteps[coord.q]
                    assert coord.t >= last_q_timestep, \
                        f"Past timesteps found for codebook={coord.q} at step {s}"
                    q_timesteps[coord.q] = coord.t
                assert len(qs) == len(seq_coords), \
                    f"Multiple entries for same codebook at step {s}"

    @property
    def num_sequence_steps(self):
        return len(self.layout) - 1

    @property
    def max_delay(self):
        max_t_in_seq_coords = 0
        for seq_coords in self.layout[1:]:
            for coords in seq_coords:
                max_t_in_seq_coords = max(max_t_in_seq_coords, coords.t + 1)
        return max_t_in_seq_coords - self.timesteps

    @property
    def valid_layout(self):
        valid_step = len(self.layout) - self.max_delay
        return self.layout[:valid_step]

    def starts_with_special_token(self):
        return self.layout[0] == []

    def get_sequence_coords_with_timestep(self, t: int, q: tp.Optional[int] = None):
        assert t <= self.timesteps
        if q is not None:
            assert q <= self.n_q
        coords = []
        for s, seq_codes in enumerate(self.layout):
            for code in seq_codes:
                if code.t == t and (q is None or code.q == q):
                    coords.append((s, code))
        return coords

    def get_steps_with_timestep(self, t: int, q: tp.Optional[int] = None) -> tp.List[int]:
        return [step for step, coords in self.get_sequence_coords_with_timestep(t, q)]

    def get_first_step_with_timesteps(self, t: int, q: tp.Optional[int] = None) -> tp.Optional[int]:
        steps = self.get_steps_with_timestep(t, q)
        return steps[0] if len(steps) > 0 else None

    def _build_pattern_sequence_scatter_indexes(
        self, timesteps: int, n_q: int, keep_only_valid_steps: bool
    ) -> tp.Tuple[mx.array, mx.array]:
        assert n_q == self.n_q
        assert timesteps <= self.timesteps
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout

        indexes = np.zeros((n_q, len(ref_layout)), dtype=np.int64)
        mask = np.zeros((n_q, len(ref_layout)), dtype=np.bool_)
        indexes[:] = n_q * timesteps  # special token index

        for s, sequence_coords in enumerate(ref_layout):
            for coords in sequence_coords:
                if coords.t < timesteps:
                    indexes[coords.q, s] = coords.t + coords.q * timesteps
                    mask[coords.q, s] = True

        return mx.array(indexes), mx.array(mask)

    def build_pattern_sequence(self, z: mx.array, special_token: int,
                               keep_only_valid_steps: bool = False):
        """Build interleaved sequence from multi-codebook input.

        Args:
            z: ``[B, K, T]`` input tensor.
            special_token: Token to fill non-pattern positions.
        Returns:
            Tuple of (values, indexes, mask).
        """
        B, K, T = z.shape
        indexes, mask = self._build_pattern_sequence_scatter_indexes(
            T, K, keep_only_valid_steps=keep_only_valid_steps)

        z = z.reshape(B, -1)  # [B, K*T]
        # Append special token
        z = mx.concatenate([z, mx.full((B, 1), special_token, dtype=z.dtype)], axis=1)

        flat_indexes = indexes.reshape(-1)  # [K*S]
        values = z[:, flat_indexes]  # [B, K*S]
        values = values.reshape(B, K, indexes.shape[-1])
        return values, indexes, mask

    def _build_reverted_sequence_scatter_indexes(
        self, sequence_steps: int, n_q: int,
        keep_only_valid_steps: bool = False,
        is_model_output: bool = False,
    ) -> tp.Tuple[mx.array, mx.array]:
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
        timesteps = self.timesteps
        assert n_q == self.n_q
        assert sequence_steps <= len(ref_layout)

        if is_model_output and self.starts_with_special_token():
            ref_layout = ref_layout[1:]

        indexes = np.zeros((n_q, timesteps), dtype=np.int64)
        mask = np.zeros((n_q, timesteps), dtype=np.bool_)
        indexes[:] = n_q * sequence_steps

        for s, sequence_codes in enumerate(ref_layout):
            if s < sequence_steps:
                for code in sequence_codes:
                    if code.t < timesteps:
                        indexes[code.q, code.t] = s + code.q * sequence_steps
                        mask[code.q, code.t] = True

        return mx.array(indexes), mx.array(mask)

    def revert_pattern_sequence(self, s: mx.array, special_token: int,
                                keep_only_valid_steps: bool = False):
        """Revert interleaved sequence back to multi-codebook format.

        Args:
            s: ``[B, K, S]`` interleaved tensor.
            special_token: Fill value for invalid positions.
        Returns:
            Tuple of (values, indexes, mask).
        """
        B, K, S = s.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S, K, keep_only_valid_steps, is_model_output=False)

        s = s.reshape(B, -1)
        s = mx.concatenate([s, mx.full((B, 1), special_token, dtype=s.dtype)], axis=1)

        flat_indexes = indexes.reshape(-1)
        values = s[:, flat_indexes]
        values = values.reshape(B, K, indexes.shape[-1])
        return values, indexes, mask

    def revert_pattern_logits(self, logits: mx.array, special_token: float,
                              keep_only_valid_steps: bool = False):
        """Revert model logits from interleaved to original sequence layout.

        Args:
            logits: ``[B, card, K, S]`` logits tensor.
            special_token: Fill value.
        Returns:
            Tuple of (values, indexes, mask).
        """
        B, card, K, S = logits.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S, K, keep_only_valid_steps, is_model_output=True)

        logits = logits.reshape(B, card, -1)
        logits = mx.concatenate(
            [logits, mx.full((B, card, 1), special_token, dtype=logits.dtype)],
            axis=-1)

        flat_indexes = indexes.reshape(-1)
        values = logits[:, :, flat_indexes]
        values = values.reshape(B, card, K, indexes.shape[-1])
        return values, indexes, mask


# ── Pattern Providers ────────────────────────────────────────────────────────

class CodebooksPatternProvider(ABC):
    """Abstract base for codebook interleaving patterns."""

    def __init__(self, n_q: int, cached: bool = True):
        assert n_q > 0
        self.n_q = n_q
        self.get_pattern = lru_cache(100)(self.get_pattern)  # type: ignore

    @abstractmethod
    def get_pattern(self, timesteps: int) -> Pattern:
        raise NotImplementedError()


class DelayedPatternProvider(CodebooksPatternProvider):
    """Delayed pattern: each codebook is delayed by an increasing offset."""

    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None,
                 flatten_first: int = 0, empty_initial: int = 0):
        super().__init__(n_q)
        if delays is None:
            delays = list(range(n_q))
        self.delays = delays
        self.flatten_first = flatten_first
        self.empty_initial = empty_initial
        assert len(self.delays) == self.n_q
        assert sorted(self.delays) == self.delays

    def get_pattern(self, timesteps: int) -> Pattern:
        omit_special_token = self.empty_initial < 0
        out: PatternLayout = [] if omit_special_token else [[]]
        max_delay = max(self.delays)
        if self.empty_initial:
            out += [[] for _ in range(self.empty_initial)]
        if self.flatten_first:
            for t in range(min(timesteps, self.flatten_first)):
                for q in range(self.n_q):
                    out.append([LayoutCoord(t, q)])
        for t in range(self.flatten_first, timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= self.flatten_first:
                    v.append(LayoutCoord(t_for_q, q))
            out.append(v)
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


class ParallelPatternProvider(DelayedPatternProvider):
    """Parallel pattern: no delay between codebooks."""

    def __init__(self, n_q: int, empty_initial: int = 0):
        super().__init__(n_q, [0] * n_q, empty_initial=empty_initial)


class UnrolledPatternProvider(CodebooksPatternProvider):
    """Unrolled codebook pattern with configurable flattening and delays."""

    FlattenedCodebook = namedtuple('FlattenedCodebook', ['codebooks', 'delay'])

    def __init__(self, n_q: int, flattening: tp.Optional[tp.List[int]] = None,
                 delays: tp.Optional[tp.List[int]] = None):
        super().__init__(n_q)
        if flattening is None:
            flattening = list(range(n_q))
        if delays is None:
            delays = [0] * n_q
        assert len(flattening) == n_q
        assert len(delays) == n_q
        assert sorted(flattening) == flattening
        assert sorted(delays) == delays
        self._flattened_codebooks = self._build_flattened_codebooks(delays, flattening)
        self.max_delay = max(delays)

    def _build_flattened_codebooks(self, delays, flattening):
        flattened_codebooks: dict = {}
        for q, (inner_step, delay) in enumerate(zip(flattening, delays)):
            if inner_step not in flattened_codebooks:
                flat_codebook = UnrolledPatternProvider.FlattenedCodebook(
                    codebooks=[q], delay=delay)
            else:
                flat_codebook = flattened_codebooks[inner_step]
                assert flat_codebook.delay == delay
                flat_codebook.codebooks.append(q)
            flattened_codebooks[inner_step] = flat_codebook
        return flattened_codebooks

    @property
    def _num_inner_steps(self):
        return max(self._flattened_codebooks.keys()) + 1

    def num_virtual_steps(self, timesteps: int) -> int:
        return timesteps * self._num_inner_steps + 1

    def get_pattern(self, timesteps: int) -> Pattern:
        indexed_out: list = [(-1, [])]
        max_timesteps = timesteps + self.max_delay
        for t in range(max_timesteps):
            for step in range(self._num_inner_steps):
                if step in self._flattened_codebooks:
                    step_codebooks = self._flattened_codebooks[step]
                    t_for_q = t + step_codebooks.delay
                    coords = [LayoutCoord(t, q) for q in step_codebooks.codebooks]
                    if t_for_q < max_timesteps and t < max_timesteps:
                        indexed_out.append((t_for_q, coords))
                else:
                    indexed_out.append((t, []))
        out = [coords for _, coords in sorted(indexed_out)]
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


class CoarseFirstPattern(CodebooksPatternProvider):
    """Coarse-first pattern: generate all coarse codebook first, then fine."""

    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None):
        super().__init__(n_q)
        if delays is None:
            delays = [0] * (n_q - 1)
        self.delays = delays
        assert len(self.delays) == self.n_q - 1
        assert sorted(self.delays) == self.delays

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        for t in range(timesteps):
            out.append([LayoutCoord(t, 0)])
        max_delay = max(self.delays)
        for t in range(timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= 0:
                    v.append(LayoutCoord(t_for_q, q + 1))
            out.append(v)
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


class MusicLMPattern(CodebooksPatternProvider):
    """MusicLM-style pattern with grouped codebooks."""

    def __init__(self, n_q: int, group_by: int = 2):
        super().__init__(n_q)
        self.group_by = group_by

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        for offset in range(0, self.n_q, self.group_by):
            for t in range(timesteps):
                for q in range(offset, offset + self.group_by):
                    out.append([LayoutCoord(t, q)])
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)
