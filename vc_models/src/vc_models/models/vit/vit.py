# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# adapted from:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import os
from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from vc_models.models.vit import model_utils
from timm.models.vision_transformer import resize_pos_embed


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self, global_pool=False, use_cls=True, mask_ratio=None, del_head=True, **kwargs
    ):
        super(VisionTransformer, self).__init__(**kwargs)
        if global_pool:
            self.classifier_feature = "global_pool"
        elif use_cls:
            self.classifier_feature = "use_cls_token"
        else:
            self.classifier_feature = "reshape_embedding"

        if del_head:
            del self.head  # don't use prediction head

        if self.classifier_feature == "global_pool":
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        if self.classifier_feature == "reshape_embedding":
            self.final_spatial = int(self.patch_embed.num_patches**0.5)
            self.embed_dim = (
                self.patch_embed.grid_size[0],
                self.patch_embed.grid_size[1],
                kwargs["embed_dim"],
            )

        self.mask_ratio = mask_ratio

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def handle_outcome(self, x):
        if self.classifier_feature == "global_pool":
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        elif self.classifier_feature == "use_cls_token":
            x = self.norm(x)
            outcome = x[:, 0]  # use cls token
        elif self.classifier_feature == "reshape_embedding":
            x = self.norm(x)
            outcome = reshape_embedding(
                x[:, 1:]
            )  # remove cls token and reshape embedding
        else:
            raise NotImplementedError

        return outcome

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if self.mask_ratio is not None:
            x, _, _ = self.random_masking(x, mask_ratio=self.mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)

        x = self.blocks(x)
        return self.handle_outcome(x)

    def forward(self, x):
        return self.forward_features(x)


class ClipVisionTransformer(VisionTransformer):
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat(
            [
                self.cls_token.squeeze()
                + torch.zeros(B, 1, x.shape[-1], device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed.squeeze().to(x.dtype)
        x = self.norm_pre(x)

        x = self.blocks(x)
        return self.handle_outcome(x)


def reshape_embedding(x):
    N, L, D = x.shape
    H = W = int(L**0.5)
    x = x.reshape(N, H, W, D)
    x = torch.einsum("nhwd->ndhw", x)
    return x


def vit_small_patch16(**kwargs):
    """ViT small as defined in the DeiT paper."""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def clip_vit_base_patch16(**kwargs):
    model = ClipVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # CLIP-specific:
        pre_norm=True,
        num_classes=512,
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def load_mae_encoder(model, checkpoint_path=None):
    if checkpoint_path is None:
        return model
    else:
        model_utils.download_model_if_needed(checkpoint_path)

    if not os.path.isabs(checkpoint_path):
        model_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..')
        checkpoint_path = os.path.join(model_base_dir,checkpoint_path)
        
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    if state_dict["pos_embed"].shape != model.pos_embed.shape:
        state_dict["pos_embed"] = resize_pos_embed(
            state_dict["pos_embed"],
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )

    # filter out keys with name decoder or mask_token
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if "decoder" not in k and "mask_token" not in k
    }

    if model.classifier_feature == "global_pool":
        # remove layer that start with norm
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("norm")}
        # add fc_norm in the state dict from the model
        state_dict["fc_norm.weight"] = model.fc_norm.weight
        state_dict["fc_norm.bias"] = model.fc_norm.bias

    model.load_state_dict(state_dict)
    return model


def load_contrastive_vit(model, checkpoint_path=None, state_dict_key="state_dict"):
    if checkpoint_path is None:
        return model

    old_state_dict = torch.load(checkpoint_path, map_location="cpu")[state_dict_key]
    state_dict = {}
    for k in list(old_state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith("module.base_encoder") and not k.startswith(
            "module.base_encoder.head"
        ):
            # remove prefix
            state_dict[k[len("module.base_encoder.") :]] = old_state_dict[k]
        # delete renamed or unused k
        del old_state_dict[k]

    if model.classifier_feature == "global_pool":
        # remove layer that start with norm
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("norm")}
        # add fc_norm in the state dict from the model
        state_dict["fc_norm.weight"] = model.fc_norm.weight
        state_dict["fc_norm.bias"] = model.fc_norm.bias

    if state_dict["pos_embed"].shape != model.pos_embed.shape:
        state_dict["pos_embed"] = resize_pos_embed(
            state_dict["pos_embed"],
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )

    model.load_state_dict(state_dict)
    return model
