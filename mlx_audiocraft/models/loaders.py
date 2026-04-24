# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Model loading from HuggingFace/local checkpoints.

import logging
import os
from pathlib import Path
import typing as tp

import mlx.core as mx
import numpy as np

from omegaconf import OmegaConf, DictConfig

from . import builders
from .encodec import CompressionModel, HFEncodecCompressionModel
from .lm import LMModel
from ..utils.weight_convert import convert_encodec_weights, convert_lm_weights

logger = logging.getLogger(__name__)


def get_audiocraft_cache_dir() -> tp.Optional[str]:
    return os.environ.get('AUDIOCRAFT_CACHE_DIR', None)


def _get_state_dict(
    file_or_url_or_id: tp.Union[Path, str],
    filename: tp.Optional[str] = None,
    cache_dir: tp.Optional[str] = None,
) -> dict:
    """Load a PyTorch state dict from file, URL, or HuggingFace Hub."""
    import torch
    from huggingface_hub import hf_hub_download

    if cache_dir is None:
        cache_dir = get_audiocraft_cache_dir()

    file_or_url_or_id = str(file_or_url_or_id)

    if os.path.isfile(file_or_url_or_id):
        return torch.load(file_or_url_or_id, map_location='cpu',
                          weights_only=False)

    if os.path.isdir(file_or_url_or_id):
        file_path = f"{file_or_url_or_id}/{filename}"
        return torch.load(file_path, map_location='cpu',
                          weights_only=False)

    if file_or_url_or_id.startswith('https://'):
        return torch.hub.load_state_dict_from_url(
            file_or_url_or_id, map_location='cpu', check_hash=True)

    # HuggingFace Hub
    assert filename is not None, \
        "filename is required when loading from HuggingFace Hub"
    file_path = hf_hub_download(
        repo_id=file_or_url_or_id,
        filename=filename,
        cache_dir=cache_dir,
    )
    return torch.load(file_path, map_location='cpu', weights_only=False)


def _delete_param(cfg: DictConfig, full_name: str):
    """Delete a parameter from a nested config."""
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)


def _load_weights_into_model(model, weights: dict, strict: bool = False):
    """Load converted numpy weights into an MLX model.

    Converts flat key-value dict to list of (key, mx.array) pairs and uses
    MLX's load_weights, which correctly handles list indexing via dot notation.
    """
    pairs = []
    for key, value in weights.items():
        if isinstance(value, np.ndarray):
            pairs.append((key, mx.array(value)))
        elif isinstance(value, mx.array):
            pairs.append((key, value))
        else:
            pairs.append((key, mx.array(np.array(value))))

    if strict:
        model.load_weights(pairs, strict=True)
    else:
        # Load what we can; log mismatches
        try:
            model.load_weights(pairs, strict=False)
        except Exception as e:
            logger.warning(f"Weight loading had issues: {e}")
            # Fallback: load one by one
            _load_weights_one_by_one(model, pairs)

    logger.info(f"Loaded {len(pairs)} weight tensors")


def _load_weights_one_by_one(model, pairs):
    """Fallback: navigate model tree manually and set each weight."""
    loaded = 0
    for key, value in pairs:
        try:
            parts = key.split('.')
            obj = model
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            loaded += 1
        except Exception as e:
            logger.debug(f"Could not set {key}: {e}")
    logger.info(f"Manually loaded {loaded}/{len(pairs)} weights")


def load_compression_model(
    file_or_url_or_id: tp.Union[Path, str],
    cache_dir: tp.Optional[str] = None,
) -> CompressionModel:
    """Load a compression model (EnCodec) from checkpoint.

    Args:
        file_or_url_or_id: Path, URL, or HuggingFace repo ID.
        cache_dir: Cache directory for downloads.
    Returns:
        Loaded compression model for MLX inference.
    """
    pkg = _get_state_dict(
        file_or_url_or_id, filename='compression_state_dict.bin',
        cache_dir=cache_dir)

    if 'pretrained' in pkg:
        # Stereo/large models reference a pretrained HF EnCodec repo
        # (e.g. facebook/encodec_32khz) that uses HuggingFace format.
        pretrained_id = pkg['pretrained']
        logger.info(f"Compression model references pretrained: {pretrained_id}")
        return load_hf_compression_model(pretrained_id)

    cfg = OmegaConf.create(pkg['xp.cfg'])
    model = builders.get_compression_model(cfg)

    # Convert and load weights
    state_dict = pkg.get('best_state', {})
    converted = convert_encodec_weights(state_dict)
    _load_weights_into_model(model, converted)

    logger.info(f"Loaded compression model from {file_or_url_or_id}")
    return model


def load_lm_model(
    file_or_url_or_id: tp.Union[Path, str],
    cache_dir: tp.Optional[str] = None,
) -> LMModel:
    """Load a language model from checkpoint.

    Args:
        file_or_url_or_id: Path, URL, or HuggingFace repo ID.
        cache_dir: Cache directory for downloads.
    Returns:
        Loaded language model for MLX inference.
    """
    pkg = _get_state_dict(
        file_or_url_or_id, filename='state_dict.bin',
        cache_dir=cache_dir)

    cfg = OmegaConf.create(pkg['xp.cfg'])

    # Clean up config for inference
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    model = builders.get_lm_model(cfg)

    # Convert and load weights
    state_dict = pkg.get('best_state', {})
    converted = convert_lm_weights(state_dict)

    # Separate condition_provider weights (loaded into PyTorch T5)
    # from LM weights (loaded into MLX model)
    lm_weights = {}
    cond_weights = {}
    for key, value in converted.items():
        if key.startswith('condition_provider.conditioners.'):
            # These are T5/conditioner weights - handle separately
            cond_key = key.replace(
                'condition_provider.conditioners.', '')
            cond_weights[cond_key] = value
        else:
            lm_weights[key] = value

    _load_weights_into_model(model, lm_weights)

    # Load condition provider output projection weights
    # (T5 model weights are loaded from HuggingFace directly)
    for cond_name, conditioner in model.condition_provider.conditioners.items():
        prefix = f"{cond_name}."
        cond_specific = {
            k[len(prefix):]: v for k, v in cond_weights.items()
            if k.startswith(prefix)
        }
        if cond_specific:
            # Only load output_proj weights (T5 backbone loaded separately)
            proj_weights = {
                k: v for k, v in cond_specific.items()
                if k.startswith('output_proj.')
            }
            if proj_weights:
                _load_weights_into_model(conditioner, proj_weights)
                logger.info(
                    f"Loaded {len(proj_weights)} output_proj weights "
                    f"for conditioner '{cond_name}'")

    # Store config for max_duration inference
    model.cfg = cfg

    logger.info(f"Loaded LM model from {file_or_url_or_id}")
    return model


def load_hf_compression_model(pretrained_id: str) -> CompressionModel:
    """Load compression model from a HuggingFace transformers pretrained model.

    Used for models like ``facebook/encodec_32khz`` that are only available
    in HuggingFace format (model.safetensors + config.json).
    The model runs in PyTorch; MLX conversion happens at encode/decode boundaries.
    """
    import torch
    from transformers import EncodecModel as HFEncodecModel

    logger.info(f"Loading HF EnCodec model: {pretrained_id}")
    hf_model = HFEncodecModel.from_pretrained(pretrained_id)
    hf_model.eval()

    model = HFEncodecCompressionModel(hf_model)
    logger.info(
        f"Loaded HF compression model: sr={model.sample_rate}, "
        f"channels={model.channels}, codebooks={model.total_codebooks}, "
        f"frame_rate={model.frame_rate}")
    return model
