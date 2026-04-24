# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – EnCodec compression model (inference only).

from abc import ABC, abstractmethod
import logging
import math
import typing as tp

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..quantization.vq import ResidualVectorQuantizer
from ..modules.seanet import SEANetEncoder, SEANetDecoder

logger = logging.getLogger(__name__)


class CompressionModel(ABC, nn.Module):
    """Base API for compression models used as audio tokenizers."""

    @abstractmethod
    def encode(self, x: mx.array) -> tp.Tuple[mx.array, tp.Optional[mx.array]]:
        ...

    @abstractmethod
    def decode(self, codes: mx.array, scale: tp.Optional[mx.array] = None) -> mx.array:
        ...

    @abstractmethod
    def decode_latent(self, codes: mx.array) -> mx.array:
        ...

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def cardinality(self) -> int: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int: ...

    @abstractmethod
    def set_num_codebooks(self, n: int): ...


class EncodecModel(CompressionModel):
    """Encodec model operating on raw waveform (inference only).

    Args:
        encoder: SEANet encoder.
        decoder: SEANet decoder.
        quantizer: Residual vector quantizer.
        frame_rate: Frame rate for the latent representation.
        sample_rate: Audio sample rate.
        channels: Number of audio channels.
        renormalize: Whether to renormalize audio.
    """

    def __init__(self, encoder: SEANetEncoder, decoder: SEANetDecoder,
                 quantizer: ResidualVectorQuantizer,
                 frame_rate: int, sample_rate: int, channels: int,
                 causal: bool = False, renormalize: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.renormalize = renormalize
        self.causal = causal

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def total_codebooks(self) -> int:
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self) -> int:
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self) -> int:
        return self.quantizer.bins

    def preprocess(self, x: mx.array) -> tp.Tuple[mx.array, tp.Optional[mx.array]]:
        """Preprocess audio, optionally renormalizing."""
        if self.renormalize:
            mono = mx.mean(x, axis=1, keepdims=True)
            volume = mx.sqrt(mx.mean(mono * mono, axis=2, keepdims=True))
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.reshape(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(self, x: mx.array,
                    scale: tp.Optional[mx.array] = None) -> mx.array:
        if scale is not None:
            x = x * scale.reshape(-1, 1, 1)
        return x

    def encode(self, x: mx.array) -> tp.Tuple[mx.array, tp.Optional[mx.array]]:
        """Encode audio to codebook indices.

        Args:
            x: ``[B, C, T]`` audio tensor.
        Returns:
            (codes, scale) where codes is ``[B, K, T']``.
        """
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb)
        return codes, scale

    def decode(self, codes: mx.array,
               scale: tp.Optional[mx.array] = None) -> mx.array:
        """Decode codebook indices to audio.

        Args:
            codes: ``[B, K, T']`` codebook indices.
            scale: Optional renormalization scale.
        Returns:
            ``[B, C, T]`` reconstructed audio.
        """
        emb = self.decode_latent(codes)
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        return out

    def decode_latent(self, codes: mx.array) -> mx.array:
        """Decode from discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)


class HFEncodecCompressionModel(CompressionModel):
    """Wrapper around HuggingFace Encodec (runs in PyTorch, converts at boundaries).

    Used for pretrained models like ``facebook/encodec_32khz`` that are only
    available in HuggingFace format.  The model runs in PyTorch on CPU;
    MLX ↔ torch conversion happens at encode/decode boundaries.
    """

    def __init__(self, hf_model):
        super().__init__()
        # Store with underscore prefix so MLX doesn't try to track it
        self._hf_model = hf_model

        # Compute possible num_codebooks from bandwidth config
        bws = hf_model.config.target_bandwidths
        hop_length = int(np.prod(hf_model.config.upsampling_ratios))
        fr = hf_model.config.sampling_rate / hop_length
        card = hf_model.config.codebook_size
        self._possible_num_codebooks = [
            int(bw * 1000 / (fr * math.log2(card))) for bw in bws
        ]
        self._num_codebooks = max(self._possible_num_codebooks)

    @property
    def channels(self) -> int:
        return self._hf_model.config.audio_channels

    @property
    def frame_rate(self) -> float:
        hop_length = int(np.prod(self._hf_model.config.upsampling_ratios))
        return self._hf_model.config.sampling_rate / hop_length

    @property
    def sample_rate(self) -> int:
        return self._hf_model.config.sampling_rate

    @property
    def cardinality(self) -> int:
        return self._hf_model.config.codebook_size

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def total_codebooks(self) -> int:
        return max(self._possible_num_codebooks)

    def set_num_codebooks(self, n: int):
        self._num_codebooks = n

    def encode(self, x: mx.array) -> tp.Tuple[mx.array, tp.Optional[mx.array]]:
        """Encode audio via HuggingFace EnCodec (PyTorch)."""
        import torch

        x_torch = torch.from_numpy(np.array(x, copy=False)).float()
        bandwidth_index = self._possible_num_codebooks.index(self._num_codebooks)
        bandwidth = self._hf_model.config.target_bandwidths[bandwidth_index]

        with torch.no_grad():
            res = self._hf_model.encode(x_torch, None, bandwidth)

        codes = res[0][0]  # first chunk
        scale = res[1][0] if res[1][0] is not None else None

        codes_mx = mx.array(codes.numpy())
        scale_mx = mx.array(scale.numpy()) if scale is not None else None
        return codes_mx, scale_mx

    def decode(self, codes: mx.array,
               scale: tp.Optional[mx.array] = None) -> mx.array:
        """Decode codes via HuggingFace EnCodec (PyTorch)."""
        import torch

        codes_torch = torch.from_numpy(np.array(codes, copy=False)).long()
        if scale is not None:
            scales = [torch.from_numpy(np.array(scale, copy=False)).float()]
        else:
            scales = [None]

        with torch.no_grad():
            # HF expects [N_chunks, B, K, T]
            out = self._hf_model.decode(codes_torch[None], scales)

        return mx.array(out[0].numpy())

    def decode_latent(self, codes: mx.array) -> mx.array:
        """Decode from discrete codes to continuous latent space."""
        import torch

        codes_torch = torch.from_numpy(np.array(codes, copy=False)).long()
        with torch.no_grad():
            out = self._hf_model.quantizer.decode(codes_torch.transpose(0, 1))
        return mx.array(out.numpy())


class InterleaveStereoCompressionModel(CompressionModel):
    """Wraps a mono CompressionModel to support stereo via interleaving.

    Left and right channels are encoded independently, and their codebooks
    are interleaved: ``[B, 2*K, T']``.
    """

    def __init__(self, model: CompressionModel):
        super().__init__()
        self.model = model

    @property
    def frame_rate(self) -> float:
        return self.model.frame_rate

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def channels(self) -> int:
        return 2

    @property
    def total_codebooks(self) -> int:
        return self.model.total_codebooks * 2

    @property
    def num_codebooks(self) -> int:
        return self.model.num_codebooks * 2

    def set_num_codebooks(self, n: int):
        assert n % 2 == 0
        self.model.set_num_codebooks(n // 2)

    @property
    def cardinality(self) -> int:
        return self.model.cardinality

    def encode(self, x: mx.array) -> tp.Tuple[mx.array, tp.Optional[mx.array]]:
        """Encode stereo: ``[B, 2, T]`` -> ``[B, 2K, T']``."""
        B, C, T = x.shape
        assert C == 2
        # Encode left and right independently
        left = x[:, :1, :]   # [B, 1, T]
        right = x[:, 1:, :]  # [B, 1, T]
        codes_left, scale_left = self.model.encode(left)
        codes_right, scale_right = self.model.encode(right)

        # Interleave: [left_q0, right_q0, left_q1, right_q1, ...]
        B, K, T_frames = codes_left.shape
        codes = mx.zeros((B, 2 * K, T_frames), dtype=codes_left.dtype)
        codes = codes.at[:, 0::2, :].add(codes_left)
        codes = codes.at[:, 1::2, :].add(codes_right)

        scale = scale_left  # Use left scale (or None)
        return codes, scale

    def decode(self, codes: mx.array,
               scale: tp.Optional[mx.array] = None) -> mx.array:
        """Decode stereo: ``[B, 2K, T']`` -> ``[B, 2, T]``."""
        B, K2, T_frames = codes.shape
        assert K2 % 2 == 0

        codes_left = codes[:, 0::2, :]   # [B, K, T']
        codes_right = codes[:, 1::2, :]  # [B, K, T']

        audio_left = self.model.decode(codes_left, scale)
        audio_right = self.model.decode(codes_right, scale)

        return mx.concatenate([audio_left, audio_right], axis=1)  # [B, 2, T]

    def decode_latent(self, codes: mx.array) -> mx.array:
        B, K2, T_frames = codes.shape
        codes_left = codes[:, 0::2, :]
        codes_right = codes[:, 1::2, :]
        latent_left = self.model.decode_latent(codes_left)
        latent_right = self.model.decode_latent(codes_right)
        return mx.concatenate([latent_left, latent_right], axis=1)
