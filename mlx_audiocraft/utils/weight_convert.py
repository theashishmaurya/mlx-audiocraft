# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# PyTorch → MLX weight conversion for AudioCraft models.

import logging
import re
import typing as tp

import numpy as np

logger = logging.getLogger(__name__)


def fold_weight_norm(state_dict: dict) -> dict:
    """Fold weight_norm parameters (weight_g + weight_v) into a single weight.

    weight = weight_g * weight_v / ||weight_v||_2
    """
    import torch

    folded = {}
    wn_keys = set()

    for key in list(state_dict.keys()):
        if key.endswith('.weight_g'):
            base = key[:-len('.weight_g')]
            wv_key = base + '.weight_v'
            if wv_key in state_dict:
                weight_g = state_dict[key]
                weight_v = state_dict[wv_key]
                # Compute norm over all dims except dim 0 (output channels)
                dims = list(range(1, weight_v.dim()))
                norm = torch.norm(weight_v, p=2, dim=dims, keepdim=True)
                weight = weight_g * weight_v / (norm + 1e-12)
                folded[base + '.weight'] = weight
                wn_keys.add(key)
                wn_keys.add(wv_key)
                logger.debug(f"Folded weight_norm: {key} + {wv_key} -> {base}.weight")

    new_state = {}
    for key, value in state_dict.items():
        if key not in wn_keys:
            new_state[key] = value
    new_state.update(folded)
    return new_state


def rename_lm_keys(state_dict: dict) -> dict:
    """Rename PyTorch MHA keys to match MLX model structure.

    - ``self_attn.in_proj_weight`` → ``self_attn.in_proj.weight``
    - ``self_attn.in_proj_bias``   → ``self_attn.in_proj.bias``
    - ``cross_attention.xxx``      → ``cross_attention_module.xxx``
    """
    renamed = {}
    for key, value in state_dict.items():
        new_key = key
        # in_proj_weight/bias → in_proj.weight/bias (raw Parameter → nn.Linear)
        new_key = new_key.replace('.in_proj_weight', '.in_proj.weight')
        new_key = new_key.replace('.in_proj_bias', '.in_proj.bias')
        # cross_attention → cross_attention_module (different attribute name)
        new_key = new_key.replace('.cross_attention.', '.cross_attention_module.')
        if new_key != key:
            logger.debug(f"Renamed LM key: {key} -> {new_key}")
        renamed[new_key] = value
    return renamed


def convert_lstm_keys(state_dict: dict) -> dict:
    """Convert PyTorch LSTM keys to MLX LSTM structure.

    PyTorch multi-layer LSTM stores per-layer weights as::

        lstm.weight_ih_l{N}  →  lstm_layers.{N}.Wx
        lstm.weight_hh_l{N}  →  lstm_layers.{N}.Wh
        lstm.bias_ih_l{N} + lstm.bias_hh_l{N}  →  lstm_layers.{N}.bias

    MLX LSTM has a single bias (PyTorch has two, summed at runtime).
    """
    # Collect LSTM keys grouped by base path
    lstm_params: tp.Dict[str, tp.Dict[str, tp.Dict[int, tp.Any]]] = {}
    lstm_key_set = set()

    pattern = re.compile(r'^(.+)\.lstm\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)$')

    for key in state_dict.keys():
        m = pattern.match(key)
        if m:
            base = m.group(1)
            param_type = m.group(2)
            layer_idx = int(m.group(3))
            if base not in lstm_params:
                lstm_params[base] = {}
            if param_type not in lstm_params[base]:
                lstm_params[base][param_type] = {}
            lstm_params[base][param_type][layer_idx] = state_dict[key]
            lstm_key_set.add(key)

    if not lstm_params:
        return state_dict

    # Build new state dict
    result = {k: v for k, v in state_dict.items() if k not in lstm_key_set}

    for base, params in lstm_params.items():
        all_layers = set()
        for param_dict in params.values():
            all_layers.update(param_dict.keys())

        for layer_idx in sorted(all_layers):
            prefix = f"{base}.lstm_layers.{layer_idx}"

            if 'weight_ih' in params and layer_idx in params['weight_ih']:
                result[f"{prefix}.Wx"] = params['weight_ih'][layer_idx]
                logger.debug(f"LSTM: {base}.lstm.weight_ih_l{layer_idx} -> {prefix}.Wx")

            if 'weight_hh' in params and layer_idx in params['weight_hh']:
                result[f"{prefix}.Wh"] = params['weight_hh'][layer_idx]
                logger.debug(f"LSTM: {base}.lstm.weight_hh_l{layer_idx} -> {prefix}.Wh")

            # Merge the two biases (PyTorch adds them during forward, MLX has one)
            bias_ih = params.get('bias_ih', {}).get(layer_idx)
            bias_hh = params.get('bias_hh', {}).get(layer_idx)
            if bias_ih is not None and bias_hh is not None:
                result[f"{prefix}.bias"] = bias_ih + bias_hh
                logger.debug(f"LSTM: merged biases -> {prefix}.bias")
            elif bias_ih is not None:
                result[f"{prefix}.bias"] = bias_ih
            elif bias_hh is not None:
                result[f"{prefix}.bias"] = bias_hh

    return result


def filter_training_only_keys(state_dict: dict) -> dict:
    """Remove training-only keys that have no corresponding parameter in inference models."""
    skip_suffixes = [
        '.embed_avg',          # EMA codebook tracking
        '.cluster_size',       # EMA codebook tracking
        '.inited',             # Initialization flag
        '.num_batches_tracked',  # BatchNorm tracking
    ]
    result = {}
    for key, value in state_dict.items():
        if any(key.endswith(s) for s in skip_suffixes):
            logger.debug(f"Skipping training-only key: {key}")
            continue
        result[key] = value
    return result


def transpose_conv_weights(state_dict: dict) -> dict:
    """Transpose Conv1d/ConvTranspose1d weights from PyTorch to MLX layout.

    PyTorch Conv1d:          [C_out, C_in, K]
    MLX Conv1d:              [C_out, K, C_in]

    PyTorch ConvTranspose1d: [C_in, C_out, K]
    MLX ConvTranspose1d:     [C_out, K, C_in]
    """
    result = {}
    for key, value in state_dict.items():
        if hasattr(value, 'numpy'):
            arr = value.cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            result[key] = value
            continue

        if arr.ndim == 3 and key.endswith('.weight'):
            if '.convtr.' in key:
                # ConvTranspose1d: [C_in, C_out, K] → [C_out, K, C_in]
                arr = np.transpose(arr, (1, 2, 0))
                logger.debug(f"Transposed ConvTranspose1d weight: {key}")
            elif '.conv.' in key:
                # Conv1d: [C_out, C_in, K] → [C_out, K, C_in]
                arr = np.transpose(arr, (0, 2, 1))
                logger.debug(f"Transposed Conv1d weight: {key}")

        result[key] = arr
    return result


def to_numpy(state_dict: dict, dtype=np.float32) -> dict:
    """Convert all tensors to numpy arrays, casting to float32 for inference.

    AudioCraft checkpoints store weights in float16 (FSDP training). We need
    float32 for stable inference without autocast.
    """
    result = {}
    for key, value in state_dict.items():
        if hasattr(value, 'numpy'):
            arr = value.cpu().float().numpy()  # .float() → float32
            result[key] = arr
        elif isinstance(value, np.ndarray):
            result[key] = value.astype(dtype) if value.dtype != dtype else value
        else:
            # Skip non-tensor values
            logger.debug(f"Skipping non-tensor: {key} ({type(value)})")
            continue
    return result


def convert_encodec_weights(state_dict: dict) -> dict:
    """Convert EnCodec PyTorch state dict to MLX format.

    Pipeline:
    1. Fold weight_norm (weight_g + weight_v → weight)
    2. Convert LSTM keys (PyTorch multi-layer → MLX per-layer)
    3. Filter training-only keys (embed_avg, cluster_size, inited)
    4. Convert to numpy
    5. Transpose conv weights to MLX layout

    Note: We do NOT remove NormConv nesting because our MLX model
    preserves the same structure (StreamableConv1d.conv → NormConv1d.conv → Conv1d).
    """
    state_dict = fold_weight_norm(state_dict)
    state_dict = convert_lstm_keys(state_dict)
    state_dict = filter_training_only_keys(state_dict)
    # Rename _codebook → codebook (MLX ignores _-prefixed attributes)
    state_dict = {k.replace('._codebook.', '.codebook.'): v
                  for k, v in state_dict.items()}
    state_dict = to_numpy(state_dict)
    state_dict = transpose_conv_weights(state_dict)
    return state_dict


def convert_lm_weights(state_dict: dict) -> dict:
    """Convert LM PyTorch state dict to MLX format.

    Pipeline:
    1. Fold weight_norm (shouldn't be any, but just in case)
    2. Filter training-only keys
    3. Rename MHA keys (in_proj_weight → in_proj.weight, cross_attention → cross_attention_module)
    4. Convert to numpy
    """
    state_dict = fold_weight_norm(state_dict)

    # Remove keys we don't need for inference
    skip_prefixes = [
        'cfg_dropout.',   # training-only dropout
        'att_dropout.',   # training-only attribute dropout
    ]

    filtered = {}
    for key, value in state_dict.items():
        if any(key.startswith(p) for p in skip_prefixes):
            continue
        filtered[key] = value

    state_dict = filter_training_only_keys(filtered)
    state_dict = rename_lm_keys(state_dict)
    state_dict = to_numpy(state_dict)

    return state_dict


def convert_full_checkpoint(
    lm_state_dict: dict,
    compression_state_dict: dict
) -> tp.Tuple[dict, dict]:
    """Convert both LM and compression model weights."""
    lm_weights = convert_lm_weights(lm_state_dict)
    compression_weights = convert_encodec_weights(compression_state_dict)
    return lm_weights, compression_weights
