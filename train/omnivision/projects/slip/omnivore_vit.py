# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Code modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py # NOQA
and https://github.com/facebookresearch/deit/blob/main/models.py by Matthew
Leavitt (ito@fb.com, matthew.l.leavitt@gmail.com) and Vedanuj Goswami
(vedanuj@fb.com).

FIXME: DO NOT OPEN SOURCE THIS - we're missing proper attributions!
"""

# FIXME: (kalyan) Hacky way to setup VISSL
import sys

sys.path.append("/private/home/kalyanv/omnivision/projects")


import copy
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import List

import hydra
import torch
import torch.nn as nn
from timm.models.layers import lecun_normal_
from vissl.config import AttrDict
from vissl.models.model_helpers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = nn.functional.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = nn.functional.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        non_skip_wt=1.0,
        non_skip_wt_learnable=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        if non_skip_wt_learnable is False:
            self.non_skip_wt = non_skip_wt
        else:
            self.non_skip_wt = nn.Parameter((torch.ones(1) * non_skip_wt).squeeze())

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) * self.non_skip_wt
        x = x + self.drop_path(self.mlp(self.norm2(x))) * self.non_skip_wt
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbedConv(nn.Module):
    def __init__(self, conv_param_list, img_size=224, patch_size=16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches

        layers = []
        for idx, k in enumerate(conv_param_list):
            conv = nn.Conv2d(
                k["input_channels"],
                k["output_channels"],
                kernel_size=k["kernel_size"],
                stride=k["stride"],
                padding=k["padding"],
                bias=k["bias"],
            )
            layers.append(conv)
            if idx != len(conv_param_list) - 1:
                if k["norm"] == "bn":
                    norm = nn.BatchNorm2d(k["output_channels"])
                    layers.append(norm)
                elif k["norm"] == "lnfp32":
                    norm = Fp32GroupNorm(1, k["output_channels"])
                    layers.append(norm)
                elif k["norm"] == "ln":
                    norm = nn.GroupNorm(1, k["output_channels"])
                    layers.append(norm)
                if k["act"] == "relu":
                    act = nn.ReLU(inplace=True)
                    layers.append(act)
                elif k["act"] == "gelu":
                    act = nn.GELU()
                    layers.append(act)
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbedGeneric(nn.Module):
    """
    PatchEmbed from Hydra
    """

    def __init__(self, cfg_list):
        super().__init__()
        proj_stem = [hydra.utils.instantiate(x, _convert_="all") for x in cfg_list]
        if len(proj_stem) > 1:
            self.proj = nn.Sequential(*proj_stem)
        else:
            # Special case to be able to load pre-trained models that were
            # trained with a standard stem
            self.proj = proj_stem[0]

    def forward(self, x):
        x = self.proj(x)
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """
    Vision transformer. Adding stochastic depth makes it a DeiT.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()

        assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"
        trunk_config = copy.deepcopy(model_config.TRUNK.VISION_TRANSFORMERS)

        logging.info("Building model: Vision Transformer from yaml config")

        img_size = trunk_config.IMAGE_SIZE
        patch_size = trunk_config.PATCH_SIZE
        in_chans = trunk_config.INPUT_CHANNELS
        embed_dim = trunk_config.HIDDEN_DIM
        depth = trunk_config.NUM_LAYERS
        num_heads = trunk_config.NUM_HEADS
        mlp_ratio = trunk_config.MLP_RATIO
        qkv_bias = trunk_config.QKV_BIAS
        qk_scale = trunk_config.QK_SCALE
        drop_rate = trunk_config.DROPOUT_RATE
        attn_drop_rate = trunk_config.ATTENTION_DROPOUT_RATE
        drop_path_rate = trunk_config.DROP_PATH_RATE
        hybrid_backbone_string = None
        use_prelogits = trunk_config.USE_PRELOGITS
        representation_size = trunk_config.REPRESENTATION_SIZE
        force_cast_ln_fp32 = trunk_config.FORCE_CAST_LN_FP32
        depth_mode = trunk_config.DEPTH_MODE
        classifier_feature = trunk_config.CLASSIFIER
        self.patch_drop_min_patches = trunk_config.get("PATCH_DROP_MIN_PATCHES", -1)
        self.patch_drop_max_patches = trunk_config.get("PATCH_DROP_MAX_PATCHES", -1)
        self.patch_drop_at_eval = trunk_config.get("PATCH_DROP_AT_EVAL", False)

        # SkipLam implemented in LvVit (https://github.com/zihangJiang/TokenLabeling/blob/main/models/lvvit.py)
        # note that this is implemented as an inverse to the `skip_lam` parameter in LvVIT
        # so `skip_lam = 2.0` => `non_skip_wt = 1/2.0`
        non_skip_wt = trunk_config.NON_SKIP_WEIGHT
        non_skip_wt_learnable = trunk_config.NON_SKIP_WEIGHT_LEARNABLE

        # TODO Implement hybrid backbones
        if "HYBRID" in trunk_config.keys():
            hybrid_backbone_string = trunk_config.HYBRID
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        if force_cast_ln_fp32:
            norm_layer = partial(Fp32LayerNorm, eps=1e-6)

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        patch_embed_type = trunk_config.PATCH_EMBED_TYPE
        assert classifier_feature in ["cls_token", "global_pool"]
        self.classifier_feature = classifier_feature

        self.depth_mode = depth_mode
        depth_chans = None
        assert in_chans == 3, "Only 3 channels supported"
        if depth_mode is not None:
            msg = f"Using depth mode {depth_mode}, embed_type: {patch_embed_type}"
            logging.info(msg)
            assert depth_mode in ["separate_d_tokens", "summed_rgb_d_tokens", "rgbd"]
            if depth_mode in ["separate_d_tokens", "summed_rgb_d_tokens"]:
                depth_chans = 1
            else:
                assert depth_mode == "rgbd"
                depth_chans = 4
        if patch_embed_type == "linear":
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            if depth_chans:
                self.depth_patch_embed = PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=depth_chans,
                    embed_dim=embed_dim,
                )
        elif patch_embed_type == "conv":
            self.patch_embed = PatchEmbedConv(
                conv_param_list=trunk_config.PATCH_EMBED_PARAMS_LIST,
                img_size=img_size,
                patch_size=patch_size,
            )
            if depth_chans:
                conv_param_list = copy.deepcopy(trunk_config.PATH_EMBED_LIST)
                conv_param_list[0]["input_channels"] = depth_chans
                self.depth_patch_embed = PatchEmbedConv(
                    conv_param_list=conv_param_list,
                    img_size=img_size,
                    patch_size=patch_size,
                )
        elif patch_embed_type == "generic":
            if depth_chans:
                raise ValueError(
                    "depth_mode is unsupported for generic patch embedding"
                )
            self.patch_embed = PatchEmbedGeneric(trunk_config.PATCH_EMBED_PARAMS_LIST)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
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
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    non_skip_wt=non_skip_wt,
                    non_skip_wt_learnable=non_skip_wt_learnable,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits
        # representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        self.pre_logits = nn.Identity()
        if use_prelogits:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module, name: str = ""):
        # NOTE conv was left to pytorch default in timm's original init
        if isinstance(module, nn.Linear):
            if name.startswith("pre_logits"):
                lecun_normal_(module.weight)
                nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(
            module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, Fp32LayerNorm)
        ):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def patch_drop(self, x, npatch_per_img, patch_start_idx=1, npatch_to_keep=None):
        """
        Randomly drop patches from the input
        Input:
            - x: B x N x C
        Returns:
            - y: B x N' x C where N' is sampled from [self.patch_drop_min_patches, self.patch_drop_max_patches]
        """
        if (
            self.patch_drop_min_patches < 0
            or self.patch_drop_min_patches == npatch_per_img
            or (npatch_to_keep is not None and npatch_to_keep < 0)
        ):
            return x

        # typically we do not drop patches at test time.
        # controlled by a flag `patch_drop_at_eval`
        # we may want to drop patches in eval mode for self-supervised teachers
        if self.training is False and self.patch_drop_at_eval is False:
            return x

        rnd_inds = [
            torch.randperm(npatch_per_img, device=x.device) for _ in range(x.shape[0])
        ]

        if npatch_to_keep is None:
            npatch_to_keep = torch.randint(
                low=self.patch_drop_min_patches,
                high=self.patch_drop_max_patches,
                size=(1,),
            ).item()
        class_tokens = x[:, :patch_start_idx, ...]
        patch_tokens = x[:, patch_start_idx:, ...]

        patch_tokens = [
            patch_tokens[i, rnd_inds[i][:npatch_to_keep]] for i in range(x.shape[0])
        ]
        patch_tokens = torch.stack(patch_tokens)
        x = torch.cat([class_tokens, patch_tokens], dim=1)
        return x

    def prepare_tokens(self, x, npatch_to_keep):
        B = x.shape[0]
        has_depth = x.shape[1] == 4

        x, x_d = self.apply_patch_embedding(x, has_depth)
        npatch_per_img = x.shape[1]

        x = self.combine_depth(x, x_d, has_depth)

        class_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole class_tokens impl from Phil Wang, thanks
        x = torch.cat((class_tokens, x), dim=1)

        pos_embed = self.get_pos_embedding(npatch_per_img, has_depth)
        x = x + pos_embed
        cls_token_idx = 0
        x = self.patch_drop(
            x,
            npatch_per_img,
            patch_start_idx=cls_token_idx + 1,
            npatch_to_keep=npatch_to_keep,
        )
        x = self.pos_drop(x)
        return x

    def apply_patch_embedding(self, x, has_depth):
        x_d = None
        if has_depth:
            if self.depth_mode in ["separate_d_tokens", "summed_rgb_d_tokens"]:
                x_rgb = x[:, :3, ...]
                x_d = x[:, 3:, ...]
                x_d = self.depth_patch_embed(x_d)
                x = self.patch_embed(x_rgb)
            else:
                x = self.depth_patch_embed(x)
        else:
            x = self.patch_embed(x)
        return x, x_d

    def get_pos_embedding(self, npatch_per_img, has_depth):
        pos_embed = self.interpolate_pos_encoding(npatch_per_img, self.pos_embed)
        if has_depth and self.depth_mode == "separate_d_tokens":
            pos_emb_without_cls = pos_embed[:, 1:, :]
            pos_emb_cls_token = pos_embed[:, :1, :]
            pos_embed = torch.cat(
                [pos_emb_cls_token, pos_emb_without_cls, pos_emb_without_cls], dim=1
            )
        return pos_embed

    def combine_depth(self, x, x_d, has_depth):
        if has_depth:
            if self.depth_mode == "separate_d_tokens":
                x = torch.cat((x, x_d), dim=1)
            if self.depth_mode == "summed_rgb_d_tokens":
                x = x + x_d
        return x

    def forward_features(self, x, npatch_to_keep):
        assert npatch_to_keep is None
        x = self.prepare_tokens(x, npatch_to_keep)

        for blk in self.blocks:
            x = blk(x)

        if self.classifier_feature == "cls_token":
            x = x[:, 0]
        elif self.classifier_feature == "global_pool":
            x = x[:, 1:, ...].mean(dim=1)
        x = self.norm(x)
        return self.pre_logits(x)

    def get_intermediate_features(self, x, names, npatch_to_keep):
        interms = []

        x = self.prepare_tokens(x, npatch_to_keep)

        # get feature from every intermediate block and apply norm
        for blk in self.blocks:
            x = blk(x)
            interms.append(self.norm(x))

        # feature names are as follows
        # blkCLS[integer] => CLS token of blk[integer]
        # concatCLS[integer] => concat of CLS token from last "integer" blocks
        # lastCLS => CLS token of last block

        output = []

        for name in names:
            if name.startswith("blkCLS"):
                v = int(name.replace("blkCLS", ""))
                output.append(interms[v][:, 0])
            elif name.startswith("concatCLS"):
                v = int(name.replace("concatCLS", ""))
                feat = torch.cat([x[:, 0] for x in interms[-v:]], dim=-1)
                output.append(feat)
            elif name == "lastCLS":
                output.append(interms[-1][:, 0])
        return output

    def forward(
        self,
        x: torch.Tensor,
        out_feat_keys: List[str] = None,
        npatch_to_keep: int = None,
    ) -> List[torch.Tensor]:
        if out_feat_keys is None or len(out_feat_keys) == 0:
            x = self.forward_features(x, npatch_to_keep)
            # x = x.unsqueeze(0) # TODO: (Kalyan) Check this!
        else:
            # we specified a feature layer name
            # Follow DINO (https://github.com/facebookresearch/dino/blob/main/eval_linear.py#L159)
            x = self.get_intermediate_features(x, out_feat_keys, npatch_to_keep)
        return x

    def interpolate_pos_encoding(self, npatch_per_img, pos_embed):
        # npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch_per_img == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = pos_embed.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=math.sqrt(npatch_per_img / N),
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()
        if layer_name in ["cls_token", "pos_embed"]:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers

    def get_num_layers(self):
        return len(self.blocks) + 1


class VisionTransformerVideo(VisionTransformer):
    """A version of Vision Transformer for videos, based on MVIT Table 3a."""

    def get_pos_embedding(self, patches_per_frame, num_frames, has_depth):
        pos_embed = self.interpolate_pos_encoding(patches_per_frame, self.pos_embed)
        num_replicas = (
            num_frames * 2
            if has_depth and self.depth_mode == "separate_d_tokens"
            else num_frames
        )
        pos_embed = torch.cat(
            (
                pos_embed[:, :1, :],  # the cls token embedding
                pos_embed[:, 1:, :].repeat(1, num_replicas, 1),
            ),
            dim=1,
        )
        return pos_embed

    def prepare_tokens(self, x):
        # TODO(rgirdhar): Add temporal position embedding. Unclear from
        # MVIT if they actually add it in the VIT baselines or not
        assert x.ndim == 5  # B, C, T, H, W
        B = x.size(0)
        num_frames = x.size(2)
        has_depth = x.shape[1] == 4
        # Stack the frames vertically to embed them into patches
        x = x.flatten(2, 3)

        x, x_d = self.apply_patch_embedding(x, has_depth)
        patches_per_frame = x.size(1) // num_frames

        x = self.combine_depth(x, x_d, has_depth)

        class_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole class_tokens impl from Phil Wang, thanks

        x = torch.cat((class_tokens, x), dim=1)

        pos_embed = self.get_pos_embedding(patches_per_frame, num_frames, has_depth)
        x = x + pos_embed
        x = self.pos_drop(x)
        return x

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if x.ndim == 4:  # Is an image
            x = x.unsqueeze(2)  # B, C, H, W -> B, C, T, H, W (T=1)
        return super().forward(x, *args, **kwargs)


class VisionTransformerVideoSepSTPos(VisionTransformerVideo):
    """A version of Vision Transformer for videos, based on MVIT Table 3a;
    with separable space time position embedding."""

    def __init__(self, model_config: AttrDict, *args, **kwargs):
        super().__init__(model_config, *args, **kwargs)
        # Create a temporal position embedding
        trunk_config = copy.deepcopy(model_config.TRUNK.VISION_TRANSFORMERS)
        if "NUM_FRAMES" not in trunk_config:
            raise ValueError("Must specify NUM_FRAMES in the config for this")
        num_frames = trunk_config.NUM_FRAMES
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, self.pos_embed.size(-1))
        )
        trunc_normal_(self.temporal_pos_embed, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay().union({"temporal_pos_embed"})

    def _apply_pos_embedding(
        self, x: torch.Tensor, patches_per_frame: int, num_frames: int
    ):
        x = super()._apply_pos_embedding(x, patches_per_frame, num_frames)
        assert num_frames == self.temporal_pos_embed.size(
            1
        ), f"TODO: Interpolated position embedding not supported temporally: Got {num_frames} frames and temporal embed is of size {self.temporal_pos_embed.size()}"
        # Now add a temporal position embedding: repeat the temporal encoding
        # for each patch. Don't add anything to the CLS token (1st element)
        x[:, 1:, :] = x[:, 1:, :] + self.temporal_pos_embed.repeat_interleave(
            patches_per_frame, dim=1
        )
        return x

    def forward(self, x, *args, **kwargs):
        if x.ndim == 4 and self.temporal_pos_embed.shape[1] > 1:
            # special case of using an image. Repeat it T times
            # This helps us use a pure video model on image datasets
            x = x.unsqueeze(2)
            x = x.repeat([1, 1, self.temporal_pos_embed.shape[1], 1, 1])
        return super().forward(x, *args, **kwargs)
