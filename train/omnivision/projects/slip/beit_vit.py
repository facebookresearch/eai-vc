# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from slip.masking_generator import MaskingGenerator
from timm.models.cait import ClassAttn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Block, PatchEmbed

# from timm.models.xcit import PositionalEncodingFourier

ASSET_DIR = (
    "/private/home/aelnouby/repos/DS-ViT/SLIP/"  # TODO: Make this a config param.
)


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        use_abs_pos_emb=True,
        use_mean_pooling=True,
        init_scale=0.001,
        cls_attn=False,
        last_layer_bn=False,
        masking=None,
        uniform=True,
        freeze_layers=None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.use_mean_pooling = use_mean_pooling

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.masking = masking
        if masking is not None:
            window_size = (img_size // patch_size, img_size // patch_size)
            self.masking_generator = MaskingGenerator(
                window_size, int(num_patches * self.masking), uniform=uniform
            )

        self.cls_attn = None
        if cls_attn:
            self.cls_attn = ClassAttn(self.embed_dim)

        self.freeze_layers = freeze_layers

        trunc_normal_(self.pos_embed, std=0.02)
        # trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        # self.head.weight.data.mul_(init_scale)
        # self.head.bias.data.mul_(init_scale)

        self.bn0 = None
        if last_layer_bn:
            self.bn0 = nn.BatchNorm1d(num_features=embed_dim, affine=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        if (self.masking is not None) and self.training:
            bool_masked_pos = torch.BoolTensor(
                np.array([self.masking_generator() for _ in range(len(x))])
            )
            bool_masked_pos = bool_masked_pos.to(x.device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

            B, seq_len, C = x.size()
            x = x[~bool_masked_pos].reshape(B, -1, C)

        batch_size, seq_len, _ = x.size()

        for blk in self.blocks:
            x = blk(x)

        tokens = x

        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x[:, 1:].mean(1)), tokens
        elif self.cls_attn is not None:
            x = self.cls_attn(x)
            return x[:, 0], tokens
        else:
            return x[:, 0], tokens

    def forward(self, x):
        x, tokens = self.forward_features(x)
        # if self.bn0 is not None:
        #     x = self.bn0(x)

        # x = self.head(x)
        return x, tokens


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            "/checkpoint/aelnouby/cvpr2022/beit_baseline/beit_imagenet_300/checkpoint-299.pth"
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def vit_small_patch16_224_50p(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        masking=0.5,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            "/checkpoint/aelnouby/cvpr2022/beit_baseline/beit_imagenet_300/checkpoint-299.pth"
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def vit_small_patch16_224_50p_block(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        masking=0.5,
        uniform=False,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            "/checkpoint/aelnouby/cvpr2022/beit_baseline/beit_imagenet_300/checkpoint-299.pth"
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def vit_base_patch16_224_50p(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        masking=0.5,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            "/checkpoint/aelnouby/cvpr2022/daet_v2/daet_contrastive_v2_base_mim_global_gatherOff_300ep/checkpoint-299.pth"
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def vit_small_patch16_224_75p(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        masking=0.75,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            "/checkpoint/aelnouby/cvpr2022/beit_baseline/beit_imagenet_300/checkpoint-299.pth"
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


# @register_model
# def vit_base_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.load('/checkpoint/aelnouby/cvpr2022/daet_v2/daet_contrastive_v2_base_mim_global_gatherOff_300ep/checkpoint-299.pth')
#         model.load_state_dict(checkpoint["model"], strict=False)
#     return model


@register_model
def dino_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mean_pooling=False,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(ASSET_DIR + "assets/dino_vitbase16_pretrain.pth")
        model.load_state_dict(checkpoint, strict=False)
    return model


@register_model
def ibot_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mean_pooling=False,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(ASSET_DIR + "assets/ibot_base.pth")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model


@register_model
def ibot_base_patch16_224_imnet21k(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mean_pooling=False,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(ASSET_DIR + "assets/ibot_base_imnet21k.pth")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model


@register_model
def mae_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mean_pooling=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(ASSET_DIR + "assets/mae_pretrain_vit_base.pth")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def mae_base_patch16_224_from_vissl(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mean_pooling=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            "/checkpoint/kalyanv/omnivision/pretrained_ckpts/vit_ckpts/mae_pretrain_vit_base.pth"
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def vit_base_24_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=24,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def mae_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mean_pooling=False,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(ASSET_DIR + "assets/mae_pretrain_vit_large.pth")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def mae_huge_patch14(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mean_pooling=False,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(ASSET_DIR + "assets/mae_pretrain_vit_huge.pth")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model
