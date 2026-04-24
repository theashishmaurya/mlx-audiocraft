# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – AudioGen: text-to-sound-effects (inference only).
#
# Architecture note
# -----------------
# AudioGen and MusicGen share the same transformer LM + EnCodec codec design.
# The only differences are:
#   • Trained on sound effects instead of music
#   • 16 kHz sample rate with 4 codebooks (MusicGen uses 32 kHz / 4–8 codebooks)
#   • No melody/chroma conditioning — text prompt only
#
# Because BaseGenModel already provides text-only _prepare_tokens_and_attributes
# and a sliding-window _generate_tokens, AudioGen just needs to:
#   1. Point get_pretrained() at the AudioGen HF checkpoints
#   2. Set SFX-appropriate generation defaults (shorter clips, tighter CFG)
#
# That's it. No model architecture code needed — it's all inherited.

import typing as tp

from .encodec import CompressionModel
from .genmodel import BaseGenModel
from .lm import LMModel


_HF_MODEL_CHECKPOINTS_MAP = {
    # Short alias → full HuggingFace repo ID
    "medium": "facebook/audiogen-medium",
}


class AudioGen(BaseGenModel):
    """AudioGen: text-to-sound-effects on Apple Silicon via MLX.

    Generates environmental sounds, foley, and audio events from text prompts.
    Runs the transformer and EnCodec decoder on MLX (Apple GPU / Neural Engine).
    The T5 text encoder runs on CPU via PyTorch (startup overhead ~3 s, then cached).

    Available models
    ----------------
    - facebook/audiogen-medium  1.5 B params, 16 kHz, ~3.2 GB download

    Example
    -------
    >>> model = AudioGen.get_pretrained("facebook/audiogen-medium")
    >>> model.set_generation_params(duration=5)
    >>> wav = model.generate(["dog barking in a park"])
    >>> # wav shape: [1, 1, samples] — batch × channels × time
    """

    def __init__(self, name: str, compression_model: CompressionModel,
                 lm: LMModel, max_duration: tp.Optional[float] = None):
        super().__init__(name, compression_model, lm, max_duration)
        # SFX are typically short — default to 5 s, not the 30 s music default
        self.set_generation_params(duration=5)

    @staticmethod
    def get_pretrained(name: str = "facebook/audiogen-medium") -> "AudioGen":
        """Load a pretrained AudioGen model from HuggingFace.

        On first call the checkpoint (~3.6 GB total) is downloaded and cached
        in ~/.cache/huggingface/. Subsequent calls are instant.

        Args:
            name: HuggingFace repo ID or short alias.
                  Supported: "facebook/audiogen-medium" (or "medium").

        Returns:
            AudioGen instance ready for generation.
        """
        import logging
        import warnings
        from .loaders import load_compression_model, load_lm_model

        logger = logging.getLogger(__name__)

        if name in _HF_MODEL_CHECKPOINTS_MAP:
            warnings.warn(
                f"Using short name '{name}'. "
                f"Use 'facebook/audiogen-{name}' instead.")
            name = _HF_MODEL_CHECKPOINTS_MAP[name]

        logger.info(f"Loading AudioGen LM: {name}")
        lm = load_lm_model(name)

        logger.info(f"Loading AudioGen compression model: {name}")
        compression_model = load_compression_model(name)

        # Extract max_duration from the embedded experiment config.
        # AudioGen-medium stores segment_duration=10 in its config.
        max_duration = None
        if hasattr(lm, 'cfg') and hasattr(lm.cfg, 'dataset'):
            max_duration = getattr(lm.cfg.dataset, 'segment_duration', None)

        logger.info(
            f"AudioGen ready — sr={compression_model.sample_rate} Hz, "
            f"channels={compression_model.channels}, "
            f"frame_rate={compression_model.frame_rate:.1f} fps, "
            f"max_duration={max_duration} s")

        return AudioGen(name, compression_model, lm, max_duration)

    def set_generation_params(
        self,
        use_sampling: bool = True,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        duration: float = 5.0,
        cfg_coef: float = 3.0,
        two_step_cfg: bool = False,
        extend_stride: float = 3.0,
    ):
        """Configure generation parameters.

        Args:
            use_sampling: Sample from the distribution (True) vs greedy decode (False).
                          Always keep True — greedy audio sounds robotic.
            top_k: Keep only the top-k most likely tokens at each step.
                   250 is a good default; lower = more conservative, higher = more varied.
            top_p: Nucleus sampling threshold (0 = disabled, rely on top_k only).
            temperature: Controls randomness. 1.0 = model default.
                         < 1.0 = more deterministic. > 1.0 = more random/creative.
            duration: Length of generated audio in seconds.
                      AudioGen-medium supports up to 10 s natively;
                      longer durations use sliding window automatically.
            cfg_coef: Classifier-free guidance strength.
                      Higher = more faithful to prompt, less varied.
                      3.0 is a good default for sound effects.
            two_step_cfg: Run conditional and unconditional passes separately.
                          Slightly slower but sometimes better quality.
            extend_stride: Overlap in seconds when generating audio longer than max_duration.
                           Must be less than max_duration.
        """
        assert extend_stride < self.max_duration, (
            f"extend_stride ({extend_stride}s) must be less than "
            f"max_duration ({self.max_duration}s)"
        )
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            "use_sampling": use_sampling,
            "temp": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "cfg_coef": cfg_coef,
            "two_step_cfg": two_step_cfg,
            "cfg_coef_beta": None,  # AudioGen doesn't use style CFG
        }

    # _prepare_tokens_and_attributes and _generate_tokens are inherited from
    # BaseGenModel unchanged — they already implement text-only conditioning
    # and sliding-window extended generation, which is exactly what AudioGen needs.
    #
    # generate(), generate_continuation(), generate_unconditional() are all
    # inherited too. No overrides needed.
