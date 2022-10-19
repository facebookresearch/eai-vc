# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T


class VIP(nn.Module):
    def __init__(
        self,
        device="cuda",
        lr=1e-4,
        embedding_dim=1024,
        backbone="vit-b",
        lweight=0.0,
        gamma=0.98,
        num_negatives=0,
    ):
        super().__init__()
        self.device = device
        self.lweight = lweight

        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.backbone = backbone
        self.num_negatives = num_negatives

        ## Distances and Metrics
        self.cs = torch.nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = Sigmoid()

        params = []
        ######################################################################## Sub Modules
        ## Visual Encoder
        if backbone == "resnet18":
            self.outdim = 512
            self.convnet = torchvision.models.resnet18(pretrained=False)
        elif backbone == "resnet34":
            self.outdim = 512
            self.convnet = torchvision.models.resnet34(pretrained=False)
        elif backbone == "resnet50":
            self.outdim = 2048
            self.convnet = torchvision.models.resnet50(pretrained=False)
        elif backbone == "vit-b":
            from eaif_models.models.vit.vit import vit_base_patch16

            self.backbone = vit_base_patch16(use_cls=True)
            self.backbone_out_dim = 768
        elif backbone == "vit-s":
            from eaif_models.models.vit.vit import vit_small_patch16

            self.backbone = vit_small_patch16(use_cls=True)
            self.backbone_out_dim = 384

        params += list(self.backbone.parameters())

        # TODO: Is this right?
        self.normlayer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.backbone_layer_norm = None
        # Adding layernorm on the output of vision backbone makes the outputs roughly gaussian.
        # This makes it well conditioned for the downstream MLP layers.
        # self.backbone_layer_norm = torch.nn.LayerNorm(self.backbone_out_dim)

        # self.embedding_dim = embedding_dim
        # self.vision_mlp_head = torch.nn.Sequential(
        #                             nn.Linear(self.backbone_out_dim, self.embedding_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(self.embedding_dim, self.embedding_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(self.embedding_dim, self.embedding_dim))
        # params += list(self.vision_mlp_head.parameters())

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(params, lr=lr)

    def forward(self, x):
        obs_shape = x.shape[1:]
        # if not already resized and cropped, then add those in preprocessing
        if obs_shape != [3, 224, 224]:
            preprocess = nn.Sequential(
                transforms.Resize(256),
                transforms.CenterCrop(224),
                self.normlayer,
            )
        else:
            preprocess = nn.Sequential(
                self.normlayer,
            )

        ## Input must be [0, 255], [3,224,224]
        x = x.float() / 255.0
        obs_p = preprocess(x)
        out = self.backbone(obs_p)
        if self.backbone_layer_norm is not None:
            out = self.backbone_layer_norm(out)
            out = self.vision_mlp_head(out)
        return out

    def sim(self, tensor1, tensor2):
        d = -torch.linalg.norm(tensor1 - tensor2, dim=-1)
        return d
