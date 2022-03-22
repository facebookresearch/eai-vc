import numpy as np
import torch
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar
from torch import nn as nn

from eai.models import resnet_gn as resnet


class VisualEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        input_channels: int = 3,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        normalize_visual_inputs: bool = True,
    ):
        super().__init__()

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(input_channels)
        else:
            self.running_mean_and_var = nn.Sequential()

        make_backbone = getattr(resnet, backbone)
        self.backbone = make_backbone(input_channels, baseplanes, ngroups)

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

        self.output_shape = (
            num_compression_channels,
            final_spatial,
            final_spatial,
        )
        self.output_size = np.prod(self.output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x
