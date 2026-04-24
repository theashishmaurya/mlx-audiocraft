# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – StreamableLSTM.

import mlx.core as mx
import mlx.nn as nn


class StreamableLSTM(nn.Module):
    """LSTM without worrying about hidden state, nor the layout of data.

    Expects input as convolutional layout ``[B, C, T]``.
    MLX LSTM expects ``[B, T, C]``, so we transpose at the boundary.
    """

    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm_layers = [nn.LSTM(dimension, dimension) for _ in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, T] -> [B, T, C] for MLX LSTM
        x = mx.transpose(x, axes=(0, 2, 1))
        y = x
        for lstm in self.lstm_layers:
            # MLX LSTM returns (hidden_states, (h_n, c_n))
            y, _ = lstm(y)
        if self.skip:
            y = y + x
        # [B, T, C] -> [B, C, T]
        y = mx.transpose(y, axes=(0, 2, 1))
        return y
