from typing import Optional

import hydra
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from habitat import logger
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar


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
    ):
        super().__init__()

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(input_channels)
        else:
            self.running_mean_and_var = nn.Sequential()

        backbone_config.defrost()
        backbone_config.transform.resize_size = image_size
        backbone_config.transform.output_size = image_size
        if use_augmentations is False:
            backbone_config.transform.jitter = False
            backbone_config.transform.shift = False
        backbone_config.freeze()

        if "resnet" in backbone_config.metadata.model:
            backbone_config.defrost()
            backbone_config.model.use_avgpool_and_flatten = False
            backbone_config.freeze()

            self.backbone, embed_dim, self.visual_transform, _ = hydra.utils.call(
                backbone_config
            )

            final_spatial_compress = 1.0 / (2**5)
            final_spatial = int(image_size * final_spatial_compress)
            self.create_compression_layer(embed_dim, final_spatial)
        elif (
            "vit" in backbone_config.metadata.model
            or "beit" in backbone_config.metadata.model
        ):

            assert (
                global_pool and use_cls
            ) is False, "Both global_pool and use_cls config param cant be 'True'"
            backbone_config.defrost()
            backbone_config.model.model.img_size = image_size
            backbone_config.model.model.global_pool = global_pool
            backbone_config.model.model.use_cls = use_cls
            backbone_config.freeze()
            self.backbone, embed_dim, self.visual_transform, _ = hydra.utils.call(
                backbone_config
            )

            # TODO: move this outside habitat codebase
            if (
                backbone_config.model.model.global_pool
                or backbone_config.model.model.use_cls
            ):
                self.compression = nn.Identity()
                self.output_size = embed_dim
            else:
                final_spatial = int(self.backbone.patch_embed.num_patches**0.5)
                self.create_compression_layer(embed_dim, final_spatial)
        else:
            raise ValueError(f"unknown backbone {backbone_config.metadata.model}")

    def create_compression_layer(self, embed_dim, final_spatial):
        after_compression_flat_size = 2048
        num_compression_channels = int(
            round(after_compression_flat_size / (final_spatial**2))
        )
        self.compression = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
        )

        output_shape = (
            num_compression_channels,
            final_spatial,
            final_spatial,
        )
        self.output_size = np.prod(output_shape)

    def forward(self, x: torch.Tensor, number_of_envs: int) -> torch.Tensor:  # type: ignore
        x = (
            x.permute(0, 3, 1, 2).float() / 255
        )  # convert channels-last to channels-first
        x = self.visual_transform(x, number_of_envs)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x
