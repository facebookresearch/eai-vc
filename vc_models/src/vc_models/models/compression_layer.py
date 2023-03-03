#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch.nn as nn


def create_compression_layer(
    embed_dim, final_spatial, after_compression_flat_size=2048
):
    num_compression_channels = int(
        round(after_compression_flat_size / (final_spatial**2))
    )
    compression = nn.Sequential(
        nn.Conv2d(
            embed_dim,
            num_compression_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(1, num_compression_channels),
        nn.ReLU(True),
        nn.Flatten(),
    )

    output_shape = (
        num_compression_channels,
        final_spatial,
        final_spatial,
    )
    output_size = np.prod(output_shape)

    return compression, output_shape, output_size
