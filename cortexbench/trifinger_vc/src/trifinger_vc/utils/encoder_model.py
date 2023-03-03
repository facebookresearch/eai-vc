#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import trifinger_vc.utils.data_utils as d_utils

import trifinger_vc.utils.model_utils as m_utils

class EncoderModel(torch.nn.Module):
    def __init__(
        self,
        pretrained_rep="r3m",
        freeze_pretrained_rep=False,
        rep_to_policy="linear_layer",
    ):
        super().__init__()

        (
            self.pretrained_rep_model,
            self.transform,
            pretrained_rep_dim,
        ) = m_utils.get_vc_model_and_transform(
            pretrained_rep, use_compression_layer=False
        )
        self.pretrained_rep = pretrained_rep
        self.pretrained_rep_dim = pretrained_rep_dim
        self.rep_to_policy = rep_to_policy

        if freeze_pretrained_rep:
            for (
                name,
                param,
            ) in self.pretrained_rep_model.named_parameters():
                param.requires_grad = False

        # this only works for ViTs
        output_rep_dim = 784
        if self.rep_to_policy == "linear_layer":
            assert (
                self.pretrained_rep_model.classifier_feature == "global_pool"
                or self.pretrained_rep_model.classifier_feature == "use_cls_token"
            )
            self.compression = nn.Sequential(
                nn.Linear(self.pretrained_rep_dim, output_rep_dim), nn.ReLU(True)
            )
        elif self.rep_to_policy == "none":
            self.compression = nn.Identity()
            output_rep_dim = pretrained_rep_dim

        # else:
        elif self.rep_to_policy == "1D_avgpool":
            assert self.pretrained_rep_model.classifier_feature == "reshape_embedding"
            self.compression = nn.AdaptiveAvgPool1d(output_rep_dim)

        self.pretrained_rep_dim = output_rep_dim

    def encode_img(self, img):
        """
        Encode img by first passing it through transform, then through model
        ** Only works for single, unbatched image **
        """

        img_preproc = self.transform(Image.fromarray(img.astype(np.uint8))).unsqueeze(0)
        device = next(self.parameters()).device
        return self.forward(img_preproc.to(device))[0].detach()

    def forward(self, input_tensor):
        x = self.pretrained_rep_model(input_tensor)
        if self.rep_to_policy == "1D_avgpool":
            N = x.shape[0]
            x = torch.einsum("ndhw->nhwd", x)
            x = x.reshape(N, -1)
        x = self.compression(x)
        return x
