#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from gym import spaces
from habitat import logger
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)

from habitat_vc.models import resnet


class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(x, start_dim=1)


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
        sem_embedding_size=4,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        if "rgb" in observation_space.spaces:
            self._frame_size = tuple(observation_space.spaces["rgb"].shape[:2])
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            # spatial_size = observation_space.spaces["rgb"].shape[:2] // 2
            spatial_size = observation_space.spaces["rgb"].shape[:2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._frame_size = tuple(observation_space.spaces["depth"].shape[:2])
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            # spatial_size = observation_space.spaces["depth"].shape[:2] // 2
            spatial_size = observation_space.spaces["depth"].shape[:2]
        else:
            self._n_input_depth = 0

        if "semantic" in observation_space.spaces:
            self._frame_size = tuple(observation_space.spaces["semantic"].shape[:2])
            self._n_input_semantics = (
                sem_embedding_size  # observation_space.spaces["semantic"].shape[2]
            )
        else:
            self._n_input_semantics = 0

        if self._frame_size == (256, 256):
            spatial_size = (128, 128)
        elif self._frame_size == (240, 320):
            spatial_size = (120, 108)
        elif self._frame_size == (480, 640):
            spatial_size = (120, 108)
        elif self._frame_size == (640, 480):
            spatial_size = (108, 120)

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = (
                self._n_input_depth + self._n_input_rgb + self._n_input_semantics
            )
            self.backbone = make_backbone(
                input_channels, baseplanes, ngroups, dropout_prob=dropout_prob
            )

            final_spatial = np.array(
                [
                    math.ceil(d * self.backbone.final_spatial_compress)
                    for d in spatial_size
                ]
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / np.prod(final_spatial))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial[0],
                final_spatial[1],
            )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth + self._n_input_semantics == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        if self._n_input_semantics > 0:
            semantic_observations = observations["semantic"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            semantic_observations = semantic_observations.permute(0, 3, 1, 2)

            cnn_input.append(semantic_observations)

        x = torch.cat(cnn_input, dim=1)
        if self._frame_size == (256, 256):
            x = F.avg_pool2d(x, 2)
        elif self._frame_size == (240, 320):
            x = F.avg_pool2d(x, (2, 3), padding=(0, 1))  # 240 x 324 -> 120 x 108
        elif self._frame_size == (480, 640):
            x = F.avg_pool2d(x, (4, 5))
        elif self._frame_size == (640, 480):
            x = F.avg_pool2d(x, (5, 4))

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=128,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_depth = observations["depth"]
        if len(obs_depth.size()) == 5:
            observations["depth"] = obs_depth.contiguous().view(
                -1, obs_depth.size(2), obs_depth.size(3), obs_depth.size(4)
            )

        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)
