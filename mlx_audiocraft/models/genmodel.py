# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Base generative model (inference only).

from abc import ABC, abstractmethod
import typing as tp

import mlx.core as mx

from .encodec import CompressionModel
from .lm import LMModel
from ..modules.conditioners import ConditioningAttributes
from ..utils.audio_utils import convert_audio


class BaseGenModel(ABC):
    """Base generative model with convenient generation API.

    Args:
        name: Model name.
        compression_model: EnCodec compression model.
        lm: Transformer language model.
        max_duration: Maximum generation duration in seconds.
    """

    def __init__(self, name: str, compression_model: CompressionModel,
                 lm: LMModel, max_duration: tp.Optional[float] = None):
        self.name = name
        self.compression_model = compression_model
        self.lm = lm

        if max_duration is None:
            max_duration = 30.0  # default
        self.max_duration: float = max_duration
        self.duration = self.max_duration
        self.extend_stride: tp.Optional[float] = None
        self.generation_params: dict = {}
        self._progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None

    @property
    def frame_rate(self) -> float:
        """AR steps per second."""
        return self.compression_model.frame_rate

    @property
    def sample_rate(self) -> int:
        """Sample rate of generated audio."""
        return self.compression_model.sample_rate

    @property
    def audio_channels(self) -> int:
        """Audio channels of generated audio."""
        return self.compression_model.channels

    def set_custom_progress_callback(
        self, progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None
    ):
        self._progress_callback = progress_callback

    @abstractmethod
    def set_generation_params(self, *args, **kwargs):
        raise NotImplementedError

    def _prepare_tokens_and_attributes(
        self,
        descriptions: tp.Sequence[tp.Optional[str]],
        prompt: tp.Optional[mx.array],
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[mx.array]]:
        """Prepare conditioning attributes and optional prompt tokens."""
        attributes = [
            ConditioningAttributes(text={'description': desc})
            for desc in descriptions
        ]

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None

        return attributes, prompt_tokens

    def generate_unconditional(self, num_samples: int,
                               progress: bool = False,
                               return_tokens: bool = False):
        """Generate samples unconditionally."""
        descriptions: tp.List[tp.Optional[str]] = [None] * num_samples
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(
            descriptions, None)
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        audio = self.generate_audio(tokens)
        if return_tokens:
            return audio, tokens
        return audio

    def generate(self, descriptions: tp.List[str],
                 progress: bool = False,
                 return_tokens: bool = False):
        """Generate samples conditioned on text descriptions."""
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(
            descriptions, None)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        audio = self.generate_audio(tokens)
        if return_tokens:
            return audio, tokens
        return audio

    def generate_continuation(self, prompt: mx.array,
                              prompt_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False,
                              return_tokens: bool = False):
        """Generate continuation from an audio prompt.

        Args:
            prompt: ``[B, C, T]`` or ``[C, T]`` audio waveform.
            prompt_sample_rate: Sample rate of the prompt.
            descriptions: Optional text descriptions.
            progress: Whether to display progress.
        """
        if prompt.ndim == 2:
            prompt = prompt[None]
        if prompt.ndim != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T]")
        prompt = convert_audio(prompt, prompt_sample_rate,
                               self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * prompt.shape[0]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(
            descriptions, prompt)
        assert prompt_tokens is not None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        audio = self.generate_audio(tokens)
        if return_tokens:
            return audio, tokens
        return audio

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[mx.array],
                         progress: bool = False) -> mx.array:
        """Generate tokens, supporting extended generation beyond max_duration."""
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration)
                             * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int,
                               tokens_to_generate: int):
            nonlocal current_gen_offset
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                self._progress_callback(generated_tokens, tokens_to_generate)
            else:
                print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}',
                      end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = _progress_callback if progress else None

        if self.duration <= self.max_duration:
            gen_tokens = self.lm.generate(
                prompt_tokens, attributes,
                callback=callback, max_gen_len=total_gen_len,
                **self.generation_params)
        else:
            # Extended generation via sliding window
            assert self.extend_stride is not None, \
                "extend_stride must be set for duration > max_duration"
            assert self.extend_stride < self.max_duration

            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            stride_tokens = int(self.frame_rate * self.extend_stride)

            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(
                    self.duration - time_offset, self.max_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)

                gen_tokens = self.lm.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=max_gen_len,
                    **self.generation_params)

                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(
                        gen_tokens[:, :, prompt_tokens.shape[-1]:])

                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = mx.concatenate(all_tokens, axis=-1)

        return gen_tokens

    def generate_audio(self, gen_tokens: mx.array) -> mx.array:
        """Decode tokens to audio waveform."""
        assert gen_tokens.ndim == 3
        gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio
