# adapted from: https://github.com/facebookresearch/mae/blob/main/models_vit.py
from functools import partial

import timm.models.beit
import torch
import torch.nn as nn


# fmt: off
class Beit(timm.models.beit.Beit):
    """ Vision Transformer w/ Distillation with support for global average pooling
    """
    def __init__(self, use_fc_norm=False, global_pool=False, use_cls=False, mask_ratio=None, use_rel_pos_bias=False, **kwargs):
        super(Beit, self).__init__(**kwargs)
        assert not (global_pool and use_cls)

        del self.head  # don't use prediction head

        self.use_fc_norm = use_fc_norm
        if self.use_fc_norm:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.global_pool = global_pool
        self.use_cls = use_cls
        self.mask_ratio = mask_ratio
        self.use_rel_pos_bias = use_rel_pos_bias

    def forward_features(self, x):
        x = self.patch_embed(x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for blk in self.blocks:
            x = blk(x, shared_rel_pos_bias=rel_pos_bias)

        if not self.use_fc_norm:
            x = self.norm(x)

        # global pooling or remove cls token
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        elif self.use_cls:
            x = x[:, 0]  # use cls token
        else:
            x = x[:, 1:]  # remove cls token

        # use fc_norm layer
        if self.use_fc_norm:
            x = self.fc_norm(x)

        return x

    def forward(self, x):
        return self.forward_features(x)


def beit_small_patch16(**kwargs):
    """beit small as defined in the BeiT paper."""
    model = Beit(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def beit_base_patch16(**kwargs):
    model = Beit(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def beit_large_patch16(**kwargs):
    model = Beit(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def beit_huge_patch14(**kwargs):
    model = Beit(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
