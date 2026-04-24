# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – MusicGen main model (inference only).

import typing as tp

import mlx.core as mx

from .encodec import CompressionModel
from .genmodel import BaseGenModel
from .lm import LMModel
from ..modules.conditioners import ConditioningAttributes, WavCondition
from ..utils.audio_utils import convert_audio


MelodyList = tp.List[tp.Optional[mx.array]]
MelodyType = tp.Union[mx.array, MelodyList]

# Backward compatible name mapping
_HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
    "style": "facebook/musicgen-style",
    "stereo-small": "facebook/musicgen-stereo-small",
    "stereo-medium": "facebook/musicgen-stereo-medium",
    "stereo-large": "facebook/musicgen-stereo-large",
    "stereo-melody": "facebook/musicgen-stereo-melody",
}


class MusicGen(BaseGenModel):
    """MusicGen main model with convenient generation API.

    Args:
        name: Model name.
        compression_model: EnCodec compression model.
        lm: Language model over discrete representations.
        max_duration: Maximum duration the model can produce per chunk.
    """

    def __init__(self, name: str, compression_model: CompressionModel,
                 lm: LMModel, max_duration: tp.Optional[float] = None):
        super().__init__(name, compression_model, lm, max_duration)
        self.set_generation_params(duration=15)  # default 15s

    @staticmethod
    def get_pretrained(name: str = 'facebook/musicgen-melody'):
        """Load a pretrained MusicGen model.

        Available models:
        - facebook/musicgen-small (300M)
        - facebook/musicgen-medium (1.5B)
        - facebook/musicgen-large (3.3B)
        - facebook/musicgen-melody (1.5B)
        - facebook/musicgen-style (1.5B)
        - facebook/musicgen-stereo-small (300M, stereo)
        - facebook/musicgen-stereo-medium (1.5B, stereo)
        - facebook/musicgen-stereo-large (3.3B, stereo)
        - facebook/musicgen-stereo-melody (1.5B, stereo)

        Also accepts short names: small, medium, large, melody, style,
        stereo-small, stereo-medium, stereo-large, stereo-melody.
        """
        import logging
        import warnings
        from .encodec import InterleaveStereoCompressionModel
        from .loaders import load_compression_model, load_lm_model

        logger = logging.getLogger(__name__)

        if name in _HF_MODEL_CHECKPOINTS_MAP:
            warnings.warn(
                f"Using deprecated short name '{name}'. "
                f"Use 'facebook/musicgen-{name}' instead.")
            name = _HF_MODEL_CHECKPOINTS_MAP[name]

        lm = load_lm_model(name)
        compression_model = load_compression_model(name)

        # Wrap in stereo if the model config specifies 2 channels
        if hasattr(lm, 'cfg'):
            cfg_channels = getattr(lm.cfg, 'channels', 1)
            if cfg_channels == 2 and compression_model.channels == 1:
                logger.info("Wrapping compression model for stereo interleaving")
                compression_model = InterleaveStereoCompressionModel(
                    compression_model)

        # Get max_duration from LM config
        max_duration = None
        if hasattr(lm, 'cfg') and hasattr(lm.cfg, 'dataset'):
            max_duration = getattr(lm.cfg.dataset, 'segment_duration', None)

        return MusicGen(name, compression_model, lm, max_duration)

    def set_generation_params(self, use_sampling: bool = True,
                              top_k: int = 250, top_p: float = 0.0,
                              temperature: float = 1.0,
                              duration: float = 30.0,
                              cfg_coef: float = 3.0,
                              cfg_coef_beta: tp.Optional[float] = None,
                              two_step_cfg: bool = False,
                              extend_stride: float = 18):
        """Set generation parameters.

        Args:
            use_sampling: Use sampling (True) or greedy decoding (False).
            top_k: K for top-k sampling.
            top_p: P for nucleus sampling (0 = disabled, uses top_k).
            temperature: Sampling temperature.
            duration: Duration of generated audio in seconds.
            cfg_coef: Classifier-free guidance coefficient.
            cfg_coef_beta: Style CFG beta (for double CFG, MusicGen-Style).
            two_step_cfg: Use two-step CFG (separate forward passes).
            extend_stride: Stride in seconds for extended generation.
        """
        assert extend_stride < self.max_duration, \
            "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
            'cfg_coef_beta': cfg_coef_beta,
        }

    def generate_with_chroma(self, descriptions: tp.List[str],
                             melody_wavs: MelodyType,
                             melody_sample_rate: int,
                             progress: bool = False,
                             return_tokens: bool = False):
        """Generate samples conditioned on text and melody.

        Args:
            descriptions: List of text descriptions.
            melody_wavs: ``[B, C, T]`` or list of ``[C, T]`` melody waveforms.
            melody_sample_rate: Sample rate of melody waveforms.
            progress: Display progress.
        """
        if isinstance(melody_wavs, mx.array):
            if melody_wavs.ndim == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.ndim != 3:
                raise ValueError("Melody wavs should have shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.ndim == 2, \
                        "Each melody should be [C, T]."

        melody_wavs = [
            convert_audio(wav, melody_sample_rate,
                          self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs
        ]

        attributes, prompt_tokens = self._prepare_tokens_and_attributes(
            descriptions=descriptions, prompt=None,
            melody_wavs=melody_wavs)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        audio = self.generate_audio(tokens)
        if return_tokens:
            return audio, tokens
        return audio

    def _prepare_tokens_and_attributes(
        self,
        descriptions: tp.Sequence[tp.Optional[str]],
        prompt: tp.Optional[mx.array],
        melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[mx.array]]:
        """Prepare conditioning attributes and optional prompt tokens."""
        attributes = [
            ConditioningAttributes(text={'description': desc})
            for desc in descriptions
        ]

        if melody_wavs is None:
            # No melody conditioning – provide null wav for models that expect it
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    wav=mx.zeros((1, 1, 1)),
                    length=mx.array([0]),
                    sample_rate=[self.sample_rate],
                    path=[None])
        else:
            assert len(melody_wavs) == len(descriptions), \
                "Number of melody wavs must match number of descriptions."
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        wav=mx.zeros((1, 1, 1)),
                        length=mx.array([0]),
                        sample_rate=[self.sample_rate],
                        path=[None])
                else:
                    attr.wav['self_wav'] = WavCondition(
                        wav=melody[None],
                        length=mx.array([melody.shape[-1]]),
                        sample_rate=[self.sample_rate],
                        path=[None])

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None

        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[mx.array],
                         progress: bool = False) -> mx.array:
        """Generate tokens with extended generation support for MusicGen.

        Handles melody/style conditioning across extended chunks.
        """
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
            # Extended generation with melody/style wav handling
            ref_wavs = [attr.wav.get('self_wav') for attr in attributes]
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            assert self.extend_stride is not None
            assert self.extend_stride < self.max_duration
            stride_tokens = int(self.frame_rate * self.extend_stride)

            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(
                    self.duration - time_offset, self.max_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)

                # Update melody/style wav for this chunk (periodic extension)
                for attr, ref_wav in zip(attributes, ref_wavs):
                    if ref_wav is None:
                        continue
                    wav_length = int(ref_wav.length[0])
                    if wav_length == 0:
                        continue
                    initial_position = int(time_offset * self.sample_rate)
                    wav_target_length = int(
                        self.max_duration * self.sample_rate)
                    positions = mx.arange(
                        initial_position,
                        initial_position + wav_target_length)
                    # Periodic extension
                    positions = positions % wav_length
                    attr.wav['self_wav'] = WavCondition(
                        wav=ref_wav.wav[..., positions],
                        length=mx.array([wav_target_length]),
                        sample_rate=[self.sample_rate],
                        path=[None],
                        seek_time=[0.])

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
