# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from github.com/openai/CLIP

import numpy as np
import slip.beit_vit as beit_vit  # noqa
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from omnivore.models.vision_transformer import Block
from slip.models import LayerNorm, ResidualAttentionBlock
from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor):
        for blk in self.resblocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


class ViTProjectionHead(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        num_layers: int,
        layer_norm_eps: float = 1e-8,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.transformer = Transformer(
            width=width,
            layers=num_layers,
            heads=num_heads,
            use_checkpoint=use_checkpoint,
        )
        self.norm = nn.LayerNorm(width, eps=layer_norm_eps)
        self.pre_logits = nn.Identity()

    def forward(self, x):
        x = self.transformer(x)
        x = x[:, 0]
        x = self.norm(x)
        return self.pre_logits(x)


class ViTProjectionHead2(nn.Module):
    def __init__(
        self,
        width: int,
        num_layers: int,
        layer_norm_eps: float = 1e-8,
        use_checkpoint: bool = False,
        attn_target=None,
        mlp_ratio=4,
        drop_rate=0.0,
        layer_scale_type=None,
        layer_scale_init_value=1e-4,
        drop_path_rate=0.0,
        drop_path_type="progressive",
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layer_norm_eps = layer_norm_eps

        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(num_layers)]
        else:
            raise ValueError(
                f"Drop path types are: [progressive, uniform]. Got {drop_path_type}."
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=width,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=self.create_layer_norm,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = self.create_layer_norm(width)
        self.pre_logits = nn.Identity()

    def create_layer_norm(self, *args, **kwargs):
        return nn.LayerNorm(*args, eps=self.layer_norm_eps, **kwargs)

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = x[:, 0]
        x = self.norm(x)
        return self.pre_logits(x)


class CLIP_PartialVision(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        vision_width: int,
        vision_model: nn.Module,
        vision_proj_head: nn.Module,
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
        self.vision_proj_head = vision_proj_head
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
            assert False, "TODO - REMOVE"
            tokens = x[1] @ self.image_projection
            x = x[0]

        x = self.vision_proj_head(x)

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
        assert (
            not use_checkpoint
        ), "Using activation is not supported in this CLIP model."

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)

        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features, self.logit_scale.exp()
