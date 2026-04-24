# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Conditioning modules (inference only).
# T5 runs via HuggingFace/PyTorch and outputs are converted to mx.arrays.

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import logging
import typing as tp

import mlx.core as mx
import mlx.nn as nn

from .streaming import StreamingModule
from .transformer import create_sin_embedding

logger = logging.getLogger(__name__)

# ── Types ────────────────────────────────────────────────────────────────────

TextCondition = tp.Optional[str]
ConditionType = tp.Tuple[mx.array, mx.array]  # (embedding, mask)


class WavCondition(tp.NamedTuple):
    wav: tp.Any  # torch.Tensor or mx.array [B, C, T]
    length: tp.Any  # torch.Tensor or mx.array
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


@dataclass
class ConditioningAttributes:
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    wav: tp.Dict[str, WavCondition] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def wav_attributes(self):
        return self.wav.keys()

    @property
    def attributes(self):
        return {
            "text": self.text_attributes,
            "wav": self.wav_attributes,
        }

    def to_flat_dict(self):
        return {
            **{f"text.{k}": v for k, v in self.text.items()},
            **{f"wav.{k}": v for k, v in self.wav.items()},
        }

    @classmethod
    def from_flat_dict(cls, x):
        out = cls()
        for k, v in x.items():
            kind, att = k.split(".")
            out[kind][att] = v
        return out


# ── Dropout / Nullification ─────────────────────────────────────────────────

def nullify_condition(condition: ConditionType, dim: int = 1) -> ConditionType:
    """Nullify a condition by zeroing it out and setting mask to 0."""
    cond, mask = condition
    B = cond.shape[0]
    # Take a single zero slice along the given dim
    slices = [slice(None)] * cond.ndim
    slices[dim] = slice(0, 1)
    out = mx.zeros_like(cond[tuple(slices)])
    new_mask = mx.zeros((B, 1), dtype=mx.int32)
    return out, new_mask


def nullify_wav(cond: WavCondition) -> WavCondition:
    """Replace a WavCondition with a null one."""
    import numpy as np
    B = cond.wav.shape[0] if hasattr(cond.wav, 'shape') else 1
    # Create a tiny null wav
    try:
        import torch
        null_wav = torch.zeros((B, 1, 1))
        null_length = torch.tensor([0] * B)
    except ImportError:
        null_wav = mx.zeros((B, 1, 1))
        null_length = mx.zeros((B,), dtype=mx.int32)
    return WavCondition(
        wav=null_wav,
        length=null_length,
        sample_rate=cond.sample_rate,
        path=[None] * B,
        seek_time=[None] * B,
    )


def dropout_condition(sample: ConditioningAttributes,
                      condition_type: str, condition: str) -> ConditioningAttributes:
    """Nullify a single attribute in a ConditioningAttributes object. Works in-place."""
    if condition_type == 'wav':
        wav_cond = sample.wav[condition]
        sample.wav[condition] = nullify_wav(wav_cond)
    else:
        sample.text[condition] = None
    return sample


class ClassifierFreeGuidanceDropout:
    """Drop all conditions with probability p. For inference, used with p=1.0
    to create null conditions for CFG."""

    def __init__(self, p: float = 0.0, seed: int = 1234):
        self.p = p

    def __call__(self, samples: tp.List[ConditioningAttributes],
                 cond_types: tp.List[str] = ["wav", "text"],
                 ) -> tp.List[ConditioningAttributes]:
        # For inference (not training), this is only called with p=1.0
        # to create null conditions, so we always drop
        if self.p >= 1.0:
            samples = deepcopy(samples)
            for condition_type in cond_types:
                for sample in samples:
                    for condition in list(sample.attributes.get(condition_type, [])):
                        dropout_condition(sample, condition_type, condition)
            return samples
        # p < 1.0 is training behavior, not used in inference
        return samples

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"


class AttributeDropout:
    """Drop individual attributes with per-attribute probabilities.
    For inference, used to selectively drop text descriptions (style CFG)."""

    def __init__(self, p: tp.Dict[str, tp.Dict[str, float]] = {},
                 active_on_eval: bool = False, seed: int = 1234):
        self.active_on_eval = active_on_eval
        self.p = {}
        for condition_type, probs in p.items():
            self.p[condition_type] = defaultdict(lambda: 0, probs)

    def __call__(self, samples: tp.List[ConditioningAttributes]
                 ) -> tp.List[ConditioningAttributes]:
        # For inference, only active when active_on_eval or when explicitly
        # called (e.g., _drop_description_condition uses p=1.0)
        samples = deepcopy(samples)
        for condition_type, ps in self.p.items():
            for condition, p in ps.items():
                if p >= 1.0:  # deterministic drop
                    for sample in samples:
                        if condition in getattr(sample, condition_type, {}):
                            dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"AttributeDropout({dict(self.p)})"


def _drop_description_condition(
    conditions: tp.List[ConditioningAttributes]
) -> tp.List[ConditioningAttributes]:
    """Drop text description but keep wav condition (for style double CFG)."""
    return AttributeDropout(p={'text': {'description': 1.0},
                               'wav': {'self_wav': 0.0}})(conditions)


# ── Base Conditioner ─────────────────────────────────────────────────────────

class BaseConditioner(nn.Module):
    """Base class for all conditioner modules."""

    def __init__(self, dim: int, output_dim: int):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        if self.output_dim > -1:
            self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs) -> tp.Any:
        raise NotImplementedError()

    def __call__(self, inputs: tp.Any) -> ConditionType:
        raise NotImplementedError()


class TextConditioner(BaseConditioner):
    """Base class for text conditioners."""
    pass


# ── T5 Conditioner ───────────────────────────────────────────────────────────

class T5Conditioner(TextConditioner):
    """T5-based text conditioner. Uses HuggingFace T5 (via PyTorch) for text
    encoding, then converts outputs to MLX arrays.

    Args:
        name: Name of the T5 model (e.g. 't5-base').
        output_dim: Output dimension for the conditioner.
        finetune: Ignored (always False for inference).
        device: Device for T5 ('cpu' or 'cuda' — T5 runs in PyTorch).
        autocast_dtype: Autocast dtype for T5 inference.
        word_dropout: Ignored for inference.
        normalize_text: Whether to normalize text.
    """
    MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base",
              "google/flan-t5-large", "google/flan-t5-xl",
              "google/flan-t5-xxl"]
    MODELS_DIMS = {
        "t5-small": 512, "t5-base": 768, "t5-large": 1024,
        "t5-3b": 1024, "t5-11b": 1024,
        "google/flan-t5-small": 512, "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024, "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(self, name: str, output_dim: int, finetune: bool = False,
                 device: str = 'cpu', autocast_dtype: tp.Optional[str] = 'float32',
                 word_dropout: float = 0., normalize_text: bool = False):
        assert name in self.MODELS, f"Unknown T5 model: {name}"
        super().__init__(self.MODELS_DIMS[name], output_dim)
        self.device = device
        self.name = name
        self.normalize_text = normalize_text

        import torch
        import warnings
        from transformers import T5Tokenizer, T5EncoderModel

        # Suppress T5 warnings during loading
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(name)
                t5 = T5EncoderModel.from_pretrained(name).eval()
            finally:
                logging.disable(previous_level)

        # Store T5 outside nn.Module parameters (not saved in checkpoints)
        self.__dict__['t5'] = t5.to(device)
        self.__dict__['_torch'] = torch

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, tp.Any]:
        """Tokenize text inputs using T5 tokenizer."""
        torch = self._torch
        entries = [xi if xi is not None else "" for xi in x]
        empty_idx = torch.LongTensor(
            [i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(
            entries, return_tensors='pt', padding=True).to(self.device)
        mask = inputs['attention_mask']
        mask[empty_idx, :] = 0
        return inputs

    def __call__(self, inputs: tp.Dict[str, tp.Any]) -> ConditionType:
        """Encode tokenized text to embeddings."""
        torch = self._torch
        mask = inputs['attention_mask']

        with torch.no_grad():
            embeds = self.t5(**inputs).last_hidden_state

        # Project to output dim (output_proj is an MLX Linear,
        # so convert to mx first)
        embeds_mx = mx.array(embeds.float().cpu().numpy())
        mask_mx = mx.array(mask.cpu().numpy())

        embeds_mx = self.output_proj(embeds_mx)
        embeds_mx = embeds_mx * mask_mx[:, :, None].astype(embeds_mx.dtype)

        return embeds_mx, mask_mx


# ── Conditioning Provider ────────────────────────────────────────────────────

class ConditioningProvider(nn.Module):
    """Prepare and provide conditions given all the supported conditioners.

    Args:
        conditioners: Dictionary of conditioner name -> conditioner module.
    """

    def __init__(self, conditioners: tp.Dict[str, BaseConditioner]):
        super().__init__()
        self.conditioners = conditioners

    @property
    def text_conditions(self):
        return [k for k, v in self.conditioners.items()
                if isinstance(v, TextConditioner)]

    @property
    def wav_conditions(self):
        # WaveformConditioner subclasses (chroma, style) would go here
        return [k for k, v in self.conditioners.items()
                if hasattr(v, '_get_wav_embedding')]

    def tokenize(self, inputs: tp.List[ConditioningAttributes]
                 ) -> tp.Dict[str, tp.Any]:
        """Tokenize all conditions from the input attributes."""
        output = {}
        text = self._collate_text(inputs)
        wavs = self._collate_wavs(inputs)

        for attribute, batch in {**text, **wavs}.items():
            if attribute in self.conditioners:
                output[attribute] = self.conditioners[attribute].tokenize(batch)

        return output

    def __call__(self, tokenized: tp.Dict[str, tp.Any]
                 ) -> tp.Dict[str, ConditionType]:
        """Compute (embedding, mask) pairs from tokenized inputs."""
        output = {}
        for attribute, inputs in tokenized.items():
            condition, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition, mask)
        return output

    def _collate_text(self, samples: tp.List[ConditioningAttributes]
                      ) -> tp.Dict[str, tp.List[tp.Optional[str]]]:
        """Collate text conditions across the batch."""
        out: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)
        for sample in samples:
            for condition in self.text_conditions:
                out[condition].append(sample.text.get(condition))
        return out

    def _collate_wavs(self, samples: tp.List[ConditioningAttributes]
                      ) -> tp.Dict[str, WavCondition]:
        """Collate wav conditions across the batch."""
        import numpy as np
        wavs_dict: tp.Dict[str, tp.List] = defaultdict(list)
        lengths_dict: tp.Dict[str, tp.List] = defaultdict(list)
        sr_dict: tp.Dict[str, tp.List] = defaultdict(list)
        paths_dict: tp.Dict[str, tp.List] = defaultdict(list)
        seek_dict: tp.Dict[str, tp.List] = defaultdict(list)
        out: tp.Dict[str, WavCondition] = {}

        for sample in samples:
            for attribute in self.wav_conditions:
                if attribute in sample.wav:
                    wav_cond = sample.wav[attribute]
                    wavs_dict[attribute].append(wav_cond.wav)
                    lengths_dict[attribute].append(wav_cond.length)
                    sr_dict[attribute].extend(wav_cond.sample_rate)
                    paths_dict[attribute].extend(wav_cond.path)
                    seek_dict[attribute].extend(wav_cond.seek_time)

        for attribute in self.wav_conditions:
            if attribute in wavs_dict:
                try:
                    import torch
                    stacked_wav = torch.cat(wavs_dict[attribute], dim=0)
                    stacked_lengths = torch.cat(lengths_dict[attribute])
                except ImportError:
                    stacked_wav = mx.concatenate(wavs_dict[attribute], axis=0)
                    stacked_lengths = mx.concatenate(lengths_dict[attribute])
                out[attribute] = WavCondition(
                    stacked_wav, stacked_lengths,
                    sr_dict[attribute],
                    paths_dict[attribute],
                    seek_dict[attribute])

        return out


# ── Condition Fuser ──────────────────────────────────────────────────────────

class ConditionFuser(StreamingModule):
    """Fuse conditions with transformer input.

    Supports:
    - 'sum': add condition embedding to input
    - 'prepend': prepend condition before input (only on first streaming step)
    - 'cross': provide as cross-attention input
    - 'input_interpolate': interpolate condition to match input length, then add
    - 'ignore': skip this condition

    Args:
        fuse2cond: Dict mapping fuse method -> list of condition names.
        cross_attention_pos_emb: Add sinusoidal pos embeddings to cross-attention.
        cross_attention_pos_emb_scale: Scale for cross-attention pos embeddings.
    """
    FUSING_METHODS = ["sum", "prepend", "cross", "ignore", "input_interpolate"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]],
                 cross_attention_pos_emb: bool = False,
                 cross_attention_pos_emb_scale: float = 1.0):
        super().__init__()
        assert all(k in self.FUSING_METHODS for k in fuse2cond.keys()), \
            f"Invalid fuse method, allowed: {self.FUSING_METHODS}"
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def __call__(
        self,
        input: mx.array,
        conditions: tp.Dict[str, ConditionType]
    ) -> tp.Tuple[mx.array, tp.Optional[mx.array]]:
        """Fuse conditions with the model input.

        Args:
            input: ``[B, T, D]`` transformer input.
            conditions: Dict of condition name -> (embedding, mask).
        Returns:
            Tuple of (fused_input, cross_attention_input or None).
        """
        B, T, _ = input.shape

        if 'offsets' in self._streaming_state:
            first_step = False
            offsets = self._streaming_state['offsets']
        else:
            first_step = True
            offsets = mx.zeros((B,), dtype=mx.int32)

        cross_attention_output = None

        for cond_type, (cond, cond_mask) in conditions.items():
            op = self.cond2fuse.get(cond_type, 'ignore')

            if op == 'sum':
                input = input + cond
            elif op == 'input_interpolate':
                # Transpose for interpolation: [B, T, D] -> [B, D, T]
                cond_t = mx.transpose(cond, axes=(0, 2, 1))
                # Simple linear interpolation to match input length
                cond_t = _interpolate_1d(cond_t, T)
                cond_t = mx.transpose(cond_t, axes=(0, 2, 1))
                input = input + cond_t
            elif op == 'prepend':
                if first_step:
                    input = mx.concatenate([cond, input], axis=1)
            elif op == 'cross':
                if cross_attention_output is not None:
                    cross_attention_output = mx.concatenate(
                        [cross_attention_output, cond], axis=1)
                else:
                    cross_attention_output = cond
            elif op == 'ignore':
                continue
            else:
                raise ValueError(f"Unknown fusing op: {op}")

        if self.cross_attention_pos_emb and cross_attention_output is not None:
            positions = mx.arange(cross_attention_output.shape[1]).reshape(1, -1, 1)
            pos_emb = create_sin_embedding(
                positions, cross_attention_output.shape[-1])
            cross_attention_output = (
                cross_attention_output
                + self.cross_attention_pos_emb_scale * pos_emb)

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return input, cross_attention_output


def _interpolate_1d(x: mx.array, target_len: int) -> mx.array:
    """Simple nearest-neighbor interpolation along last dim.
    x: [B, D, T] -> [B, D, target_len]
    """
    T = x.shape[-1]
    if T == target_len:
        return x
    # Compute source indices for each target position
    indices = mx.floor(
        mx.arange(target_len, dtype=mx.float32) * T / target_len
    ).astype(mx.int32)
    indices = mx.clip(indices, 0, T - 1)
    return x[:, :, indices]
