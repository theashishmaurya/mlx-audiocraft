# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – custom gated linear unit activations.

import typing as tp

import mlx.core as mx
import mlx.nn as nn


class CustomGLU(nn.Module):
    """Custom Gated Linear Unit activation: ``a * f(b)`` where a, b are
    the two halves of the input along *dim*.
    """
    def __init__(self, activation: nn.Module, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        assert x.shape[self.dim] % 2 == 0
        a, b = mx.split(x, 2, axis=self.dim)
        return a * self.activation(b)


class SwiGLU(CustomGLU):
    """SiLU Gated Linear Unit: ``a * SiLU(b)``."""
    def __init__(self, dim: int = -1):
        super().__init__(nn.SiLU(), dim)


class GeGLU(CustomGLU):
    """GELU Gated Linear Unit: ``a * GELU(b)``."""
    def __init__(self, dim: int = -1):
        super().__init__(nn.GELU(), dim)


class ReGLU(CustomGLU):
    """ReLU Gated Linear Unit: ``a * ReLU(b)``."""
    def __init__(self, dim: int = -1):
        super().__init__(nn.ReLU(), dim)


def get_activation_fn(
    activation: tp.Union[str, tp.Callable[[mx.array], mx.array]]
) -> tp.Union[str, tp.Callable[[mx.array], mx.array]]:
    """Map an activation string to the corresponding activation class/function."""
    if isinstance(activation, str):
        if activation == "reglu":
            return ReGLU()
        elif activation == "geglu":
            return GeGLU()
        elif activation == "swiglu":
            return SwiGLU()
    return activation
