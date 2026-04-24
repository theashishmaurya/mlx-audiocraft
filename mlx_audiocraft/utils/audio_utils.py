# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – audio I/O and conversion utilities.
# Uses soundfile/scipy for resampling instead of torchaudio/julius.

import typing as tp
from pathlib import Path

import numpy as np
import mlx.core as mx


def convert_audio_channels(wav: mx.array, channels: int) -> mx.array:
    """Convert audio to the given number of channels.

    Args:
        wav: Audio tensor of shape ``[B, C, T]`` or ``[C, T]``.
        channels: Target number of channels (1 or 2).
    """
    shape = wav.shape
    if shape[-2] == channels:
        return wav
    if channels == 1:
        # Downmix to mono: average across channels
        return mx.mean(wav, axis=-2, keepdims=True)
    if channels == 2:
        # Upmix to stereo: duplicate mono channel
        return mx.repeat(wav, repeats=2, axis=-2)
    raise ValueError(f"Unsupported number of target channels: {channels}")


def convert_audio(wav: mx.array, from_rate: int, to_rate: int,
                  to_channels: int) -> mx.array:
    """Convert audio sample rate and number of channels.

    Args:
        wav: ``[B, C, T]`` audio tensor.
        from_rate: Source sample rate.
        to_rate: Target sample rate.
        to_channels: Target number of channels.
    Returns:
        Converted audio tensor ``[B, C, T']``.
    """
    wav = convert_audio_channels(wav, to_channels)
    if from_rate != to_rate:
        wav = _resample(wav, from_rate, to_rate)
    return wav


def _resample(wav: mx.array, from_rate: int, to_rate: int) -> mx.array:
    """Resample audio using scipy (runs on CPU, converts back to MLX).

    This is used only for audio pre/post-processing, not in the neural
    network forward pass, so CPU overhead is acceptable.
    """
    try:
        from scipy.signal import resample_poly
    except ImportError:
        raise ImportError("scipy is required for resampling: pip install scipy")

    import math
    gcd = math.gcd(from_rate, to_rate)
    up = to_rate // gcd
    down = from_rate // gcd

    wav_np = np.array(wav)
    # resample_poly works on last axis
    resampled = resample_poly(wav_np, up, down, axis=-1)
    return mx.array(resampled)


def audio_read(filepath: tp.Union[str, Path],
               seek_time: float = 0.,
               duration: float = -1.,
               sample_rate: tp.Optional[int] = None) -> tp.Tuple[mx.array, int]:
    """Read an audio file and return ``(wav, sample_rate)``.

    Args:
        filepath: Path to audio file.
        seek_time: Start reading from this time (seconds).
        duration: Duration to read in seconds (-1 for full file).
        sample_rate: If provided, resample to this rate.
    Returns:
        Tuple of ``(wav, sr)`` where wav is ``[1, C, T]``.
    """
    import soundfile as sf

    info = sf.info(str(filepath))
    sr = info.samplerate

    start_frame = int(seek_time * sr) if seek_time > 0 else 0
    frames = int(duration * sr) if duration > 0 else -1

    data, sr = sf.read(str(filepath), start=start_frame, frames=frames,
                       dtype='float32', always_2d=True)
    # soundfile returns [T, C], we want [1, C, T]
    wav = mx.array(data.T)[None]

    if sample_rate is not None and sample_rate != sr:
        wav = _resample(wav, sr, sample_rate)
        sr = sample_rate

    return wav, sr


def audio_write(filepath: tp.Union[str, Path], wav: mx.array,
                sample_rate: int, format: str = 'wav') -> Path:
    """Write audio tensor to file.

    Args:
        filepath: Output path (without extension).
        wav: ``[C, T]`` or ``[B, C, T]`` audio tensor.
        sample_rate: Sample rate.
        format: Output format ('wav', 'mp3', etc.).
    Returns:
        Path to the written file.
    """
    import soundfile as sf

    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix(f'.{format}')

    wav_np = np.array(wav)
    # Remove batch dim if present
    if wav_np.ndim == 3:
        wav_np = wav_np[0]
    # [C, T] -> [T, C] for soundfile
    wav_np = wav_np.T

    sf.write(str(filepath), wav_np, sample_rate)
    return filepath
