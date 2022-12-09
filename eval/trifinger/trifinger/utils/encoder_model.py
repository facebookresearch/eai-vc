import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from eaif_models.models.compression_layer import create_compression_layer

from utils.decoder_model import DecoderModel
import utils.data_utils as d_utils

import ipdb


class EncoderModel(torch.nn.Module):
    def __init__(
        self,
        pretrained_rep="r3m",
        mlp_dim=512,
        latent_dim=50,
        freeze_pretrained_rep=False,
    ):
        super().__init__()

        (
            self.pretrained_rep_model,
            self.transform,
            pretrained_rep_dim,
        ) = d_utils.get_eaif_model_and_transform(
            pretrained_rep, use_compression_layer=False
        )
        self.pretrained_rep = pretrained_rep
        self.pretrained_rep_dim = pretrained_rep_dim

        if freeze_pretrained_rep:
            for (
                name,
                param,
            ) in self.pretrained_rep_model.named_parameters():
                param.requires_grad = False

        # this only works for ViTs
        # if self.pretrained_rep_model.global_pool or self.pretrained_rep_model.use_cls:
        #     self.compression = nn.Identity()
        # else:
        #     self.compression, _, _ = create_compression_layer(
        #         pretrained_rep_dim, self.pretrained_rep_model.final_spatial
        #     )

        if isinstance(latent_dim, int):
            linear_layers = [
                nn.Linear(pretrained_rep_dim, mlp_dim),
                nn.ELU(),
                nn.Linear(mlp_dim, mlp_dim),
                nn.ELU(),
                nn.Linear(mlp_dim, latent_dim),
            ]

            self.linear_model = nn.Sequential(*linear_layers)
        else:
            # Don't use additional linear layers
            self.linear_model = None

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
        # x = self.compression(x)
        if self.linear_model:
            x = self.linear_model(x)
        return x


class EncDecModel(torch.nn.Module):
    def __init__(
        self,
        pretrained_rep="r3m",
        mlp_dim=512,
        latent_dim=50,
        dec_dim=256,
        freeze_pretrained_rep=False,
    ):
        super().__init__()

        self.encoder_model = EncoderModel(
            pretrained_rep=pretrained_rep,
            mlp_dim=mlp_dim,
            latent_dim=latent_dim,
            freeze_pretrained_rep=freeze_pretrained_rep,
        )

        # Set encoder output dimension
        if isinstance(latent_dim, int):
            enc_out_dim = latent_dim
        else:
            # If latent_dim is None, enc_out_dim = pretrained_rep_dim
            enc_out_dim = self.encoder_model.pretrained_rep_dim

        self.decoder_model = DecoderModel(
            latent_dim=enc_out_dim,
            mlp_dim=mlp_dim,
            dec_dim=dec_dim,
        )

    def forward(self, input_tensor):

        x = self.encoder_model(input_tensor)  # [B, latent_dim]
        x = self.decoder_model(x)  # [B, 3, 64, 64]
        return x
