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
        avgpooled_image: bool = True,
        input_channels: int = 3,
        image_size: int = 128,
        normalize_visual_inputs: bool = True,
    ):
        super().__init__()
        self.avgpooled_image = avgpooled_image

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(input_channels)
        else:
            self.running_mean_and_var = nn.Sequential()

        if "resnet" in backbone_config.model_name:
            self.backbone, embed_dim, _, _ = hydra.utils.call(backbone_config)

            spatial_size = image_size
            if self.avgpooled_image:
                spatial_size = image_size // 2

            # TODO: remove dependency on final_spatial_compress
            final_spatial = int(spatial_size * self.backbone.final_spatial_compress)
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
        elif (
            "vit" in backbone_config.model_name or "beit" in backbone_config.model_name
        ):
            if self.avgpooled_image:
                image_size = image_size // 2

            backbone_config.defrost()
            backbone_config.model_config.img_size = image_size
            backbone_config.freeze()
            self.backbone, embed_dim, _, _ = hydra.utils.call(backbone_config)

            # TODO: move this outside habitat codebase
            if (
                backbone_config.model_config.global_pool
                or backbone_config.model_config.use_cls
            ):
                self.compression = nn.Identity()
                self.output_size = embed_dim
            else:
                assert backbone_config.model_config.mask_ratio == 0.0
                final_spatial = int(self.backbone.patch_embed.num_patches**0.5)
                after_compression_flat_size = 2048
                num_compression_channels = int(
                    round(after_compression_flat_size / (final_spatial**2))
                )
                self.compression = nn.Sequential(
                    ViTReshape(),
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
        else:
            raise ValueError("unknown backbone {}".format(backbone))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        if (
            self.avgpooled_image
        ):  # For compatibility with the habitat_baselines implementation
            x = F.avg_pool2d(x, 2)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class ViTReshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, L, D = x.shape
        H = W = int(L**0.5)
        x = x.reshape(N, H, W, D)
        x = torch.einsum("nhwd->ndhw", x)
        return x
