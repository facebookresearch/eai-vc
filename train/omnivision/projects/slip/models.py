# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from github.com/openai/CLIP
import contextlib
import sys
from collections import OrderedDict

import numpy as np
import slip.beit_vit as beit_vit  # noqa
import slip.losses as losses
import timm
import torch
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim):
        super().__init__()
        # self.dense = nn.Linear(embed_dim, embed_dim)
        # self.activation_fn = QuickGELU()
        # self.layer_norm = nn.LayerNorm(embed_dim)

        # self.weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        # self.bias = nn.Parameter(torch.zeros(output_dim))

        self.lm_head = nn.Linear(embed_dim, output_dim)

    def forward(self, features, masked_tokens=None, **kwargs):
        x = torch.nn.functional.dropout(features, p=0.1)
        return self.lm_head(x)

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = nn.functional.linear(x, self.weight) + self.bias
        return x


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        context_length: int,
        vocab_size: int,
        embed_dim: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()
        self.transformer_width = transformer_width
        self.context_length = context_length
        self.vocab_size = vocab_size

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer_width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class CLIPRobertaTextEncoder(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        embed_dim: int,
        transformer_width: int,
        freeze_transformer: bool = False,
        use_cls_token: bool = True,
    ):
        super().__init__()

        self.transformer = transformer
        self.use_cls_token = use_cls_token
        self.freeze_transformer = freeze_transformer
        self.transformer_width = transformer_width

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=self.transformer_width**-0.5)

        if self.freeze_transformer:
            self.transformer.eval()

    def forward(self, text):
        if self.freeze_transformer:
            context = torch.no_grad()
        else:
            context = contextlib.nullcontext()
        with context:
            x = self.transformer(text)

        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x[:, 1:].mean(dim=1)

        x = x @ self.text_projection
        return x


class CLIP_V2(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        vision_width: int,
        vision_model: nn.Module,
        # text
        text_model: nn.Module,
        freeze_vision=False,
        freeze_text=False,
        **kwargs,
    ):
        super().__init__()

        self.visual = vision_model
        self.text_model = text_model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))

        self.freeze_vision = freeze_vision
        self.freeze_text = freeze_text

        if self.freeze_vision:
            self.visual.eval()

        if self.freeze_text:
            self.text_model.eval()

        nn.init.normal_(self.image_projection, std=vision_width**-0.5)

    def encode_image(self, image, **visual_kwargs):
        if self.freeze_vision:
            context = torch.no_grad()
        else:
            context = contextlib.nullcontext()
        with context:
            x = self.visual(image, **visual_kwargs)
        x = x @ self.image_projection
        return x

    def encode_text(self, text):
        if self.freeze_text:
            context = torch.no_grad()
        else:
            context = contextlib.nullcontext()
        with context:
            text_features = self.text_model(text)
        return text_features

    def forward(self, image, text, **visual_kwargs):
        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image, **visual_kwargs)

        image_features = self.encode_image(image, **visual_kwargs)
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features, self.logit_scale.exp()


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        vision_width: int,
        vision_model: nn.Module,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        text_model: nn.Module = "scratch",
        drop_text=None,
        colbert=False,
        freeze_vision=False,
        convert_vid_to_image=False,
        **kwargs,
    ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model
        self.colbert = colbert
        self.convert_vid_to_image = convert_vid_to_image

        if "roberta" in text_model:
            self.transformer = torch.hub.load("pytorch/fairseq", "roberta.large.mnli")
            for p in self.transformer.parameters():
                p.requires_grad = False

            self.transformer.eval()
            transformer_width = 1024
        else:
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
            )

        self.text_model = text_model
        self.transformer_width = transformer_width

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.drop_text = drop_text

        self.freeze_vision = freeze_vision

        if self.freeze_vision:
            self.visual.eval()

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if "roberta" not in self.text_model:
            proj_std = (self.transformer.width**-0.5) * (
                (2 * self.transformer.layers) ** -0.5
            )
            attn_std = self.transformer.width**-0.5
            fc_std = (2 * self.transformer.width) ** -0.5
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width**-0.5)
        nn.init.normal_(self.text_projection, std=self.transformer_width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image, use_checkpoint=False):
        # Convert single frame videos to images
        #  B, C, T, H, W - >  B, C, H, W
        if self.convert_vid_to_image:
            assert image.ndim == 5 and image.shape[2] == 1
            image = image.squeeze(2)

        if self.freeze_vision:
            with torch.no_grad():
                x = self.visual(image)
        else:
            x = self.visual(image, use_checkpoint=use_checkpoint)

        if isinstance(x, list):
            x = x[0]  # TODO: Hack to handle Omnivore models that return list.

        if isinstance(x, tuple):
            tokens = x[1] @ self.image_projection
            x = x[0]

        if len(x.shape) == 3:
            x = x.squeeze(
                0
            )  # TODO: Hack to handle Omnivore VIT that return additional dimension.

        x = x @ self.image_projection

        if self.colbert and self.training:
            return x, tokens
        else:
            return x

    def encode_text(self, text):
        if self.drop_text is not None:
            for t in text:
                idx = [i for i, x in enumerate(t) if x not in [0, 49407, 49406]]
                drop_idx = np.random.choice(
                    idx, int(len(idx) * self.drop_text), replace=False
                )

                t[drop_idx] = 0

        if "roberta" in self.text_model:
            with torch.no_grad():
                x = self.transformer.extract_features(text)

            x = x[:, 0]  # .mean(dim=1)
            x = self.ln_final(x)
            x = x @ self.text_projection
            return x
        else:
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)

        tokens = x

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        if self.colbert and self.training:
            return x, tokens
        else:
            return x

    def forward(self, image, text, use_checkpoint=False):
        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image, use_checkpoint=use_checkpoint)
        image_features = self.encode_image(image, use_checkpoint=use_checkpoint)
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features, self.logit_scale.exp()


class SIMCLR(nn.Module):
    def __init__(
        self,
        # vision
        vision_width: int,
        vision_model: nn.Module,
        # ssl
        ssl_mlp_dim: int,
        ssl_emb_dim: int,
        **kwargs,
    ):
        super().__init__()

        self.vision_width = vision_width
        self.visual = vision_model

        self.image_mlp = self._build_mlp(
            in_dim=vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim
        )

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(
            OrderedDict(
                [
                    ("layer1", nn.Linear(in_dim, mlp_dim)),
                    ("bn1", nn.SyncBatchNorm(mlp_dim)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("layer2", nn.Linear(mlp_dim, mlp_dim)),
                    ("bn2", nn.SyncBatchNorm(mlp_dim)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("layer3", nn.Linear(mlp_dim, out_dim)),
                ]
            )
        )

    def encode_image(self, image):
        x = self.visual(image)

        return x

    def forward(self, aug1, aug2):
        h1 = self.visual(aug1)
        h2 = self.visual(aug2)

        aug1_embed = self.image_mlp(h1)
        aug2_embed = self.image_mlp(h2)

        return {"aug1_embed": aug1_embed, "aug2_embed": aug2_embed}


class SLIP(CLIP):
    def __init__(self, ssl_mlp_dim: int, ssl_emb_dim: int, **kwargs):
        super().__init__(**kwargs)

        self.image_mlp = self._build_mlp(
            in_dim=self.vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim
        )

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(
            OrderedDict(
                [
                    ("layer1", nn.Linear(in_dim, mlp_dim)),
                    ("bn1", nn.SyncBatchNorm(mlp_dim)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("layer2", nn.Linear(mlp_dim, mlp_dim)),
                    ("bn2", nn.SyncBatchNorm(mlp_dim)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("layer3", nn.Linear(mlp_dim, out_dim)),
                ]
            )
        )

    def forward(self, image, text, aug1, aug2):
        aug1_embed = self.image_mlp(self.visual(aug1))
        aug2_embed = self.image_mlp(self.visual(aug2))

        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),
            "aug1_embed": aug1_embed,
            "aug2_embed": aug2_embed,
        }


def get_loss(
    model,
    ssl_temp,
    ssl_scale,
    negative_threshold=None,
    temp=0.07,
    img_distill=False,
    out_dim=65536,
):
    if model.startswith("SLIP"):
        ssl_loss = losses.SIMCLRLoss(temperature=ssl_temp)
        return losses.SLIPLoss(ssl_loss, ssl_scale)
    if model.startswith("CLIP"):
        return losses.CLIPLoss(negative_threshold=negative_threshold)
    if model.startswith("SIMCLR"):
        return losses.SIMCLRLoss(temperature=ssl_temp)
    if model.startswith("CrossDistill"):
        return losses.CrossDistillationLoss(
            temp=temp, img_distill=img_distill, out_dim=out_dim
        )


def get_meter_names(model):
    if model.startswith("SLIP"):
        return ["loss", "clip_loss", "ssl_loss", "clip_acc", "ssl_acc"]
    elif model.startswith("CLIP"):
        return ["loss", "clip_loss", "clip_acc"]
    elif model.startswith("CrossDistill"):
        return ["loss", "clip_loss", "clip_acc", "cross_distill_loss"]
    else:
        return ["loss", "ssl_loss", "ssl_acc"]


@timm.models.registry.register_model
def vit_small_mocov3_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=12, **kwargs)
    model = timm.models.vision_transformer._create_vision_transformer(
        "vit_small_patch16_224", **model_kwargs
    )

    return model


def CLIP_VITS16(**kwargs):
    vision_model = timm.create_model("vit_small_mocov3_patch16_224", num_classes=0)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )

    return model


def CLIP_COLBERT_VITS16(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224", pretrained=False)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        colbert=True,
        **kwargs,
    )

    return model


def CLIP_COLBERT_50p_VITS16(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=False)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        colbert=True,
        **kwargs,
    )

    return model


def CLIP_BEITS16(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )

    return model


def CLIP_DINO_B16(**kwargs):
    vision_model = timm.create_model("dino_base_patch16_224", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        freeze_vision=False,
        **kwargs,
    )

    return model


def CLIP_MAE_B16(**kwargs):
    vision_model = timm.create_model("mae_base_patch16_224", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        freeze_vision=False,
        **kwargs,
    )

    return model


def CLIP_MAE_B16_Frozen(**kwargs):
    vision_model = timm.create_model("mae_base_patch16_224", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        freeze_vision=True,
        **kwargs,
    )

    return model


def CLIP_MAE_B16_Frozen_from_vissl(**kwargs):
    vision_model = timm.create_model("mae_base_patch16_224_from_vissl", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        freeze_vision=True,
        **kwargs,
    )

    return model


def CLIP_MAE_L16(**kwargs):
    vision_model = timm.create_model("mae_large_patch16_224", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=1024,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        freeze_vision=False,
        **kwargs,
    )

    return model


def CLIP_MAE_H16(**kwargs):
    vision_model = timm.create_model("mae_huge_patch14", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=1280,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        freeze_vision=False,
        **kwargs,
    )

    return model


def CLIP_DINO_B16_Frozen(**kwargs):
    vision_model = timm.create_model("dino_base_patch16_224", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        freeze_vision=True,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )

    return model


def CLIP_ViTS16_50pMask(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=False)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )

    return model


def CLIP_BEITS16_75pMask(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_75p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=2,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pBlockMask_TinyTextEncoder(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p_block", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=2,
        **kwargs,
    )

    return model


def CLIP_ORIGINAL_VITB32():
    try:
        import clip
    except ImportError:
        print("CLIP not found. Please install if from: https://github.com/openai/CLIP")

    model, _ = clip.load("ViT-B/32")
    return model


def CLIP_BEITBase_50pMask_TinyTextEncoder(**kwargs):
    vision_model = timm.create_model("vit_base_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=768,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=2,
        **kwargs,
    )

    return model


def CLIP_BEITS16_75pMask_TinyTextEncoder(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_75p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=2,
        **kwargs,
    )

    return model


def CLIP_BEITS16_RobertaBase(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        text_model="roberta",
        **kwargs,
    )

    return model


def CLIP_IBOT_B16(**kwargs):
    # model_kwargs = dict( **kwargs)
    # vision_model = beit_vit.vit_small_patch16_224(pretrained=True, **model_kwargs)
    vision_model = timm.create_model("ibot_base_patch16_224", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        freeze_vision=False,
        **kwargs,
    )

    return model


def CLIP_IBOT_B16_IMNET21k(**kwargs):
    # model_kwargs = dict( **kwargs)
    # vision_model = beit_vit.vit_small_patch16_224(pretrained=True, **model_kwargs)
    vision_model = timm.create_model("ibot_base_patch16_224_imnet21k", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        freeze_vision=False,
        **kwargs,
    )

    return model


def CLIP_IBOT_B16_Frozen(**kwargs):
    # model_kwargs = dict( **kwargs)
    # vision_model = beit_vit.vit_small_patch16_224(pretrained=True, **model_kwargs)
    vision_model = timm.create_model("ibot_base_patch16_224", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        freeze_vision=True,
        **kwargs,
    )

    return model


def CLIP_IBOT_B16_IMNET21k_Frozen(**kwargs):
    # model_kwargs = dict( **kwargs)
    # vision_model = beit_vit.vit_small_patch16_224(pretrained=True, **model_kwargs)
    vision_model = timm.create_model("ibot_base_patch16_224_imnet21k", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        freeze_vision=True,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_RobertaBase(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        text_model="roberta",
        **kwargs,
    )

    return model


def CLIP_VITS16_RobertaLargeFrozen(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224", pretrained=False)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=1024,
        transformer_heads=8,
        transformer_layers=12,
        text_model="roberta_frozen",
        **kwargs,
    )

    return model


def SIMCLR_VITS16(**kwargs):
    vision_model = timm.create_model("vit_small_mocov3_patch16_224", num_classes=0)
    model = SIMCLR(vision_width=384, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITS16(**kwargs):
    vision_model = timm.create_model("vit_small_mocov3_patch16_224", num_classes=0)
    model = SLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )

    return model


def CLIP_VITB16(**kwargs):
    vision_model = timm.create_model("vit_base_patch16_224", num_classes=0)
    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )

    return model


def SIMCLR_VITB16(**kwargs):
    vision_model = timm.create_model("vit_base_patch16_224", num_classes=0)
    model = SIMCLR(vision_width=768, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITB16(**kwargs):
    vision_model = timm.create_model("vit_base_patch16_224", num_classes=0)
    model = SLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )

    return model


def CLIP_VITL16(**kwargs):
    vision_model = timm.create_model("vit_large_patch16_224", num_classes=0)
    model = CLIP(
        embed_dim=512,
        vision_width=1024,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )

    return model


def SIMCLR_VITL16(**kwargs):
    vision_model = timm.create_model("vit_large_patch16_224", num_classes=0)
    model = SIMCLR(vision_width=1024, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITL16(**kwargs):
    vision_model = timm.create_model("vit_large_patch16_224", num_classes=0)
    model = SLIP(
        embed_dim=512,
        vision_width=1024,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )

    return model


# =============
def CLIP_BEITS16_Mask_TinyTextEncoder_4x768(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=4,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder_2x384(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=384,
        transformer_heads=12,
        transformer_layers=2,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder_2x768(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=2,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder_2x1024(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=1024,
        transformer_heads=16,
        transformer_layers=2,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder_4x384(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=384,
        transformer_heads=12,
        transformer_layers=4,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder_4x768(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=4,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder_4x1024(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=1024,
        transformer_heads=16,
        transformer_layers=4,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder_6x384(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=384,
        transformer_heads=12,
        transformer_layers=6,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder_6x768(**kwargs):
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=6,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder_6x1024(**kwargs):
    # model_kwargs = dict( **kwargs)
    # vision_model = beit_vit.vit_small_patch16_224(pretrained=True, **model_kwargs)
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=1024,
        transformer_heads=16,
        transformer_layers=6,
        **kwargs,
    )

    return model


# =============
def CLIP_BEITS16_50pMask_TinyTextEncoder_DropText15p(**kwargs):
    # model_kwargs = dict( **kwargs)
    # vision_model = beit_vit.vit_small_patch16_224(pretrained=True, **model_kwargs)
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=2,
        drop_text=0.15,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder_DropText30p(**kwargs):
    # model_kwargs = dict( **kwargs)
    # vision_model = beit_vit.vit_small_patch16_224(pretrained=True, **model_kwargs)
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=2,
        drop_text=0.3,
        **kwargs,
    )

    return model


def CLIP_BEITS16_50pMask_TinyTextEncoder_DropText50p(**kwargs):
    # model_kwargs = dict( **kwargs)
    # vision_model = beit_vit.vit_small_patch16_224(pretrained=True, **model_kwargs)
    vision_model = timm.create_model("vit_small_patch16_224_50p", pretrained=True)
    model = CLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=2,
        drop_text=0.5,
        **kwargs,
    )

    return model


def _make_default_omnivore_vit_cfg():
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.create(
        {
            "INPUT_TYPE": "rgb",
            "TRUNK": {
                "VISION_TRANSFORMERS": {
                    "IMAGE_SIZE": 224,
                    "PATCH_SIZE": 16,
                    "INPUT_CHANNELS": 3,
                    "HIDDEN_DIM": 384,
                    "NUM_LAYERS": 12,
                    "NUM_HEADS": 12,
                    "MLP_RATIO": 4.0,
                    "QKV_BIAS": True,
                    "QK_SCALE": False,
                    "DROPOUT_RATE": 0.1,
                    "ATTENTION_DROPOUT_RATE": 0.0,
                    "DROP_PATH_RATE": 0.0,
                    "USE_PRELOGITS": False,
                    "FORCE_CAST_LN_FP32": False,
                    "DEPTH_MODE": None,
                    "CLASSIFIER": "cls_token",
                    "PATCH_DROP_MIN_PATCHES": -1,
                    "PATCH_DROP_MAX_PATCHES": -1,
                    "PATCH_EMBED_TYPE": "linear",
                    "PATCH_DROP_AT_EVAL": False,
                    "NON_SKIP_WEIGHT": 1.0,
                    "NON_SKIP_WEIGHT_LEARNABLE": False,
                    "PATCH_EMBED_PARAMS_LIST": [],
                    "PATH_EMBED_LIST": [],
                }
            },
        }
    )
    return model_cfg


def SLIP_OMNIVORE_VITS16(**kwargs):
    from omnivore_vit import VisionTransformer

    vision_model = VisionTransformer(
        model_config=_make_default_omnivore_vit_cfg(),
        model_name="omnivore_vit_small_mocov3_patch16_224",
    )

    model = SLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )

    return model


def create_timm_finetuneModel_from_ckpt(arch, ckpt_path, num_classes=1000):
    linear_keyword = "head"

    print("=> loading checkpoint '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    visual_keyword = "module.visual."

    # rename CLIP pre-trained keys
    state_dict = checkpoint["state_dict"]
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith(visual_keyword) and not k.startswith(
            visual_keyword + linear_keyword
        ):
            # remove prefix
            state_dict[k[len(visual_keyword) :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # create model
    print("=> creating model '{}'".format(arch))
    model = timm.models.create_model(arch, num_classes=num_classes)

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {
        "%s.weight" % linear_keyword,
        "%s.bias" % linear_keyword,
    }

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ["%s.weight" % linear_keyword, "%s.bias" % linear_keyword]:
            param.requires_grad = False
    # init the fc layer
    getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
    getattr(model, linear_keyword).bias.data.zero_()

    return model


# Checkpoint utils
def _slip_clip_ckpt(ckpt_path):
    with g_pathmgr.open(ckpt_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v

    old_args = ckpt["args"]
    print("=> creating model: {}".format(old_args.model))
    model = getattr(sys.modules[__name__], old_args.model)(
        rand_embed=False,
        ssl_mlp_dim=old_args.ssl_mlp_dim,
        ssl_emb_dim=old_args.ssl_emb_dim,
    )
    model.cuda()
    model.load_state_dict(state_dict, strict=True)

    return model


class ImageEncoder(nn.Module):
    def __init__(self, visual, image_projection):
        super().__init__()
        self.visual = visual
        self.image_projection = image_projection

    def forward(self, image):
        x = self.visual(image)

        if isinstance(x, list):
            x = x[0]  # TODO: Hack to handle Omnivore models that return list.

        if isinstance(x, tuple):
            x = x[0]

        x = x @ self.image_projection

        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        token_embedding,
        positional_embedding,
        transformer,
        ln_final,
        text_projection,
    ):
        super().__init__()
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        self.ln_final = ln_final  # LayerNorm
        self.text_projection = text_projection

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class VisionTextModel(nn.Module):
    def __init__(self, vision, text, convert_vid_to_image=False):
        super().__init__()
        self.vision = vision
        self.text = text
        self.convert_vid_to_image = convert_vid_to_image

    def encode_image(self, image):
        if self.convert_vid_to_image:
            assert image.ndim == 5 and image.shape[2] == 1
            image = image.squeeze(2)
        return self.vision(image)

    def encode_text(self, text):
        return self.text(text)


def load_vision_model_from_lv_checkpoint(ckpt_path):
    model = _slip_clip_ckpt(ckpt_path)
    return ImageEncoder(model.visual, model.image_projection)


def load_text_model_from_lv_checkpoint(ckpt_path):
    model = _slip_clip_ckpt(ckpt_path)
    return TextEncoder(
        model.token_embedding,
        model.positional_embedding,
        model.transformer,
        model.ln_final,
        model.text_projection,
    )


# FIXME: (Kalyan) Once Head attacher is re-factored, remove this.
class SingleHeadWrapper(nn.Module):
    def __init__(
        self, trunk: nn.Module, head_finetune: nn.Module, finetune_trunk=True
    ) -> None:
        super().__init__()

        self.trunk = trunk
        self.finetune_trunk = finetune_trunk
        if not finetune_trunk:
            for _, param in self.trunk.named_parameters():
                param.requires_grad = False

        self.head_finetune = head_finetune

    def forward(self, *args, **kwargs):
        output = self.trunk(*args, **kwargs)
        if not self.finetune_trunk:
            output = output.detach()
        return self.head_finetune(output)


def consturct_simple_finetune_linear_head(in_features: int, out_features: int):
    head = nn.Linear(in_features, out_features)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    return head
