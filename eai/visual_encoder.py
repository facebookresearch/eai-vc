from typing import Optional
import numpy as np
import torch
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar
from torch import nn as nn
from torch.nn import functional as F

from eai.models import resnet_gn as resnet
from eai.models import vit


class VisualEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        backbone: str,
        input_channels: int = 3,
        resnet_baseplanes: int = 32,
        resnet_ngroups: int = 32,
        vit_use_fc_norm: bool = False,
        vit_global_pool: bool = False,
        vit_mask_ratio: Optional[float] = None,
        normalize_visual_inputs: bool = True,
        avgpooled_image: bool = False,
    ):
        super().__init__()
        self.avgpooled_image = avgpooled_image

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(input_channels)
        else:
            self.running_mean_and_var = nn.Sequential()

        if "resnet" in backbone:
            make_backbone = getattr(resnet, backbone)
            self.backbone = make_backbone(
                input_channels, resnet_baseplanes, resnet_ngroups
            )

            spatial_size = image_size
            if self.avgpooled_image:
                spatial_size = image_size // 2

            final_spatial = int(spatial_size * self.backbone.final_spatial_compress)
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial**2))
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

            output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )
            self.output_size = np.prod(output_shape)
        elif "vit" in backbone:
            make_backbone = getattr(vit, backbone)
            self.backbone = make_backbone(
                img_size=image_size,
                use_fc_norm=vit_use_fc_norm,
                global_pool=vit_global_pool,
                mask_ratio=vit_mask_ratio,
            )

            if self.backbone.global_pool:
                self.compression = nn.Identity()
                self.output_size = self.backbone.embed_dim
            else:
                assert self.backbone.mask_ratio is None
                final_spatial = int(self.backbone.patch_embed.num_patches**0.5)
                after_compression_flat_size = 2048
                num_compression_channels = int(
                    round(after_compression_flat_size / (final_spatial**2))
                )
                self.compression = nn.Sequential(
                    ViTReshape(),
                    nn.Conv2d(
                        self.backbone.embed_dim,
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
        if self.avgpooled_image:  # For compatibility with the habitat_baselines implementation
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
