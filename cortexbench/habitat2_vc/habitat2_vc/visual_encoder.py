#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import torch
from torch import nn as nn
from omegaconf import open_dict

from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar
from vc_models.models.compression_layer import create_compression_layer


class VisualEncoder(nn.Module):
    def __init__(
        self,
        backbone_config: str,
        input_channels: int = 3,
        image_size: int = 128,
        normalize_visual_inputs: bool = True,
        global_pool: bool = False,
        use_cls: bool = False,
        use_augmentations: bool = False,
        loaded_backbone_data=None,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.freeze_backbone = freeze_backbone
        self.is_resnet = "resnet" in backbone_config.metadata.model

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(input_channels)
        else:
            self.running_mean_and_var = nn.Sequential()

        backbone_config.transform.resize_size = image_size
        backbone_config.transform.output_size = image_size
        if use_augmentations is False:
            backbone_config.transform.jitter = False
            backbone_config.transform.shift = False

        if "resnet" in backbone_config.metadata.model:
            with open_dict(backbone_config):
                # In the case of the VIP, the fc layer is part of the model
                # so we don't use the compression layer but the fc layer + avgpool
                if "vip" in backbone_config.metadata.algo:
                    backbone_config.model.use_avgpool_and_flatten = True
                else:
                    backbone_config.model.use_avgpool_and_flatten = False

            if loaded_backbone_data is None:
                (
                    self.backbone,
                    self.embed_dim,
                    self.visual_transform,
                    _,
                ) = hydra.utils.call(backbone_config)
            else:
                (
                    self.backbone,
                    self.embed_dim,
                    self.visual_transform,
                ) = loaded_backbone_data

            if not backbone_config.model.use_avgpool_and_flatten:
                final_spatial_compress = 1.0 / (2**5)
                final_spatial = int(image_size * final_spatial_compress)
                self.compression, _, self.output_size = create_compression_layer(
                    self.embed_dim, final_spatial
                )
            else:
                self.output_size = self.embed_dim
                self.compression = nn.Sequential()

        elif (
            "vit" in backbone_config.metadata.model
            or "beit" in backbone_config.metadata.model
        ):
            assert (
                global_pool and use_cls
            ) is False, "Both global_pool and use_cls config param cant be 'True'"
            if "model" in backbone_config.model:
                model = backbone_config.model.model
            else:
                model = backbone_config.model

            with open_dict(model):
                if (
                    backbone_config.metadata.algo == "omnimae"
                    or backbone_config.metadata.algo == "tmae"
                ):
                    model.img_size = [3, image_size, image_size]
                else:
                    model.img_size = image_size

                model.global_pool = global_pool
                model.use_cls = use_cls

            if loaded_backbone_data is None:
                (
                    self.backbone,
                    self.embed_dim,
                    self.visual_transform,
                    _,
                ) = hydra.utils.call(backbone_config)
            else:
                (
                    self.backbone,
                    self.embed_dim,
                    self.visual_transform,
                ) = loaded_backbone_data

            if model.global_pool or model.use_cls:
                self.compression = nn.Identity()
                self.output_size = self.embed_dim
            else:
                self.compression, _, self.output_size = create_compression_layer(
                    self.embed_dim, self.backbone.final_spatial
                )
        else:
            raise ValueError(f"unknown backbone {backbone_config.metadata.model}")

    def get_loaded_backbone_data(self):
        return self.backbone, self.embed_dim, self.visual_transform

    def forward(
        self, x: torch.Tensor, number_of_envs: int
    ) -> torch.Tensor:  # type: ignore
        # convert channels-last to channels-first
        x = x.permute(0, 3, 1, 2).float() / 255

        # Apply visual transforms
        x = self.visual_transform(x)

        # If the transformations have normalization, do not apply running mean and var
        if "Normalize" not in str(self.visual_transform):
            x = self.running_mean_and_var(x)

        # For resnets, make sure that the model is is in eval mode and
        # that the gradients are not computed
        if self.is_resnet and self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)

        # Apply compression layer
        x = self.compression(x)
        return x
