# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Model builder functions from Hydra config.

import typing as tp

from omegaconf import OmegaConf, DictConfig

from ..modules.codebooks_patterns import (
    CodebooksPatternProvider,
    DelayedPatternProvider,
    ParallelPatternProvider,
    UnrolledPatternProvider,
    CoarseFirstPattern,
    MusicLMPattern,
)
from ..modules.conditioners import (
    BaseConditioner,
    ConditionFuser,
    ConditioningProvider,
    T5Conditioner,
)
from ..modules.seanet import SEANetEncoder, SEANetDecoder
from ..quantization.vq import ResidualVectorQuantizer
from .encodec import EncodecModel, InterleaveStereoCompressionModel, CompressionModel
from .lm import LMModel


def dict_from_config(cfg):
    """Convert an OmegaConf config to a plain dict/list, recursively."""
    if isinstance(cfg, (list, tuple)):
        return [dict_from_config(v) for v in cfg]
    if isinstance(cfg, dict):
        return {k: dict_from_config(v) for k, v in cfg.items()}
    if OmegaConf.is_config(cfg):
        if OmegaConf.is_list(cfg):
            return [dict_from_config(v) for v in cfg]
        return {k: dict_from_config(v) for k, v in cfg.items()}
    return cfg


def get_quantizer(quantizer: str, cfg: DictConfig, dimension: int):
    """Build a quantizer from config."""
    if quantizer == 'rvq':
        kwargs = dict_from_config(getattr(cfg, 'rvq'))
        kwargs['dimension'] = dimension
        return ResidualVectorQuantizer(**kwargs)
    else:
        raise KeyError(f"Unexpected quantizer: {quantizer}")


def get_encodec_autoencoder(encoder_name: str, cfg: DictConfig):
    """Build SEANet encoder/decoder from config."""
    if encoder_name == 'seanet':
        kwargs = dict_from_config(getattr(cfg, 'seanet'))
        encoder_override = kwargs.pop('encoder', {})
        decoder_override = kwargs.pop('decoder', {})
        encoder_kwargs = {**kwargs, **encoder_override}
        decoder_kwargs = {**kwargs, **decoder_override}
        encoder = SEANetEncoder(**encoder_kwargs)
        decoder = SEANetDecoder(**decoder_kwargs)
        return encoder, decoder
    else:
        raise KeyError(f"Unexpected encoder: {encoder_name}")


def get_compression_model(cfg: DictConfig) -> CompressionModel:
    """Build compression model from config."""
    if cfg.compression_model == 'encodec':
        kwargs = dict_from_config(getattr(cfg, 'encodec'))
        encoder_name = kwargs.pop('autoencoder')
        quantizer_name = kwargs.pop('quantizer')
        encoder, decoder = get_encodec_autoencoder(encoder_name, cfg)
        quantizer = get_quantizer(quantizer_name, cfg, encoder.dimension)
        frame_rate = kwargs['sample_rate'] // encoder.hop_length
        renormalize = kwargs.pop('renormalize', False)
        kwargs.pop('renorm', None)  # deprecated param
        model = EncodecModel(
            encoder, decoder, quantizer,
            frame_rate=frame_rate,
            renormalize=renormalize,
            **kwargs)

        # Wrap in stereo model if channels == 2
        if kwargs.get('channels', 1) == 2:
            model = InterleaveStereoCompressionModel(model)

        return model
    else:
        raise KeyError(f"Unexpected compression model: {cfg.compression_model}")


def get_conditioner_provider(output_dim: int, cfg: DictConfig) -> ConditioningProvider:
    """Build conditioning provider from config."""
    cond_cfg = getattr(cfg, 'conditioners', None)
    if cond_cfg is None:
        return ConditioningProvider({})

    dict_cfg = dict_from_config(cond_cfg)
    conditioners: tp.Dict[str, BaseConditioner] = {}

    # Remove args that are not conditioner definitions
    dict_cfg.pop('args', None)

    for cond_name, cond_cfg_item in dict_cfg.items():
        if not isinstance(cond_cfg_item, dict):
            continue
        model_type = cond_cfg_item.get('model', None)
        if model_type is None:
            continue

        model_args = cond_cfg_item.get(model_type, {})
        if isinstance(model_args, dict):
            model_args = dict(model_args)
        else:
            model_args = dict_from_config(model_args)

        if model_type == 't5':
            # T5 conditioner runs on CPU (or optionally on MPS via PyTorch)
            device = str(getattr(cfg, 'device', 'cpu'))
            # For MLX, we always run T5 on CPU via PyTorch
            t5_device = 'cpu'
            conditioners[str(cond_name)] = T5Conditioner(
                output_dim=output_dim, device=t5_device, **model_args)
        elif model_type == 'lut':
            # LUT conditioner (lookup table) – skipped for now
            pass
        elif model_type == 'chroma_stem':
            # ChromaStemConditioner – to be implemented
            pass
        elif model_type == 'style':
            # StyleConditioner – to be implemented
            pass
        else:
            pass  # Skip unknown conditioner types

    return ConditioningProvider(conditioners)


def get_condition_fuser(cfg: DictConfig) -> ConditionFuser:
    """Build condition fuser from config."""
    fuser_cfg = getattr(cfg, 'fuser', None)
    if fuser_cfg is None:
        return ConditionFuser(fuse2cond={'cross': ['description']})

    fuser_methods = ['sum', 'cross', 'prepend', 'ignore', 'input_interpolate']
    fuse2cond = {}
    kwargs = {}
    for k, v in fuser_cfg.items():
        if k in fuser_methods:
            fuse2cond[k] = list(v) if v else []
        else:
            kwargs[k] = v

    return ConditionFuser(fuse2cond=fuse2cond, **kwargs)


def get_codebooks_pattern_provider(
    n_q: int, cfg: DictConfig
) -> CodebooksPatternProvider:
    """Build codebooks pattern provider from config."""
    pattern_providers = {
        'parallel': ParallelPatternProvider,
        'delay': DelayedPatternProvider,
        'unroll': UnrolledPatternProvider,
        'coarse_first': CoarseFirstPattern,
        'musiclm': MusicLMPattern,
    }
    name = cfg.modeling
    kwargs = dict_from_config(cfg.get(name, {})) if hasattr(cfg, name) else {}
    klass = pattern_providers[name]
    return klass(n_q, **kwargs)


def get_lm_model(cfg: DictConfig) -> LMModel:
    """Build the transformer language model from config."""
    kwargs = dict_from_config(getattr(cfg, 'transformer_lm'))
    n_q = kwargs.pop('n_q', 8)
    q_modeling = kwargs.pop('q_modeling', None)

    codebooks_pattern_cfg = getattr(cfg, 'codebooks_pattern')
    attribute_dropout = dict_from_config(getattr(cfg, 'attribute_dropout', {}))
    cls_free_guidance = dict_from_config(getattr(cfg, 'classifier_free_guidance', {}))
    cfg_prob = cls_free_guidance.get('training_dropout', 0.0)
    cfg_coef = cls_free_guidance.get('inference_coef', 1.0)

    fuser = get_condition_fuser(cfg)
    condition_provider = get_conditioner_provider(kwargs.get('dim', 1024), cfg)

    if len(fuser.fuse2cond.get('cross', [])) > 0:
        kwargs['cross_attention'] = True

    if codebooks_pattern_cfg.modeling is None:
        assert q_modeling is not None
        codebooks_pattern_cfg = OmegaConf.create(
            {'modeling': q_modeling, 'delay': {'delays': list(range(n_q))}})

    pattern_provider = get_codebooks_pattern_provider(n_q, codebooks_pattern_cfg)

    # Extract card (codebook vocabulary size) from config
    card = kwargs.pop('card', None)
    if card is None:
        card = 1024  # default
    # Override from encodec config if available (more reliable)
    if hasattr(cfg, 'encodec') and hasattr(cfg.encodec, 'rvq'):
        card = getattr(cfg.encodec.rvq, 'bins', card)

    # Remove training-only and non-LM params from kwargs
    for param in ['dtype', 'device', 'weight_init', 'depthwise_init',
                  'zero_bias_init', 'max_seq_len',
                  'compression_model_framerate', 'segment_duration',
                  'span_len', 'q_modeling']:
        kwargs.pop(param, None)

    return LMModel(
        pattern_provider=pattern_provider,
        condition_provider=condition_provider,
        fuser=fuser,
        n_q=n_q,
        card=card,
        cfg_dropout=cfg_prob,
        cfg_coef=cfg_coef,
        attribute_dropout=attribute_dropout,
        **kwargs)
