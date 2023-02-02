from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
from omnivore.models.helpers import VerboseNNModule
from omnivore.models.vision_transformer import (
    get_sinusoid_encoding_table,
    VisionTransformer,
)
from timm.models.layers import trunc_normal_


class PatchEmbedGeneric(nn.Module):
    """
    PatchEmbed from Hydra
    """

    def __init__(self, proj_stem, norm_layer: nn.Module):
        super().__init__()

        if len(proj_stem) > 1:
            self.proj = nn.Sequential(*proj_stem)
        else:
            # Special case to be able to load pre-trained models that were
            # trained with a standard stem
            self.proj = proj_stem[0]
        self.norm_layer = norm_layer

    def get_patch_layout(self, img_size):
        with torch.no_grad():
            dummy_img = torch.zeros(
                [
                    1,
                ]
                + img_size
            )
            dummy_out = self.proj(dummy_img)
        embed_dim = dummy_out.shape[1]
        patches_layout = tuple(dummy_out.shape[2:])
        num_patches = np.prod(patches_layout)
        return patches_layout, num_patches, embed_dim

    def forward(self, x):
        x = self.proj(x)
        # B C (T) H W -> B (T)HW C
        x = x.flatten(2).transpose(1, 2)
        if self.norm_layer:
            x = self.norm_layer(x)
        return x


class SpatioTemporalPosEmbeddingHelper(VerboseNNModule):
    def __init__(
        self,
        patches_layout: List,
        num_patches: int,
        num_cls_tokens: int,
        embed_dim: int,
        learnable: bool,
    ) -> None:
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.patches_layout = patches_layout
        self.num_patches = num_patches
        self.num_tokens = num_cls_tokens + num_patches
        self.learnable = learnable
        if self.learnable:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer(
                "pos_embed", get_sinusoid_encoding_table(self.num_tokens, embed_dim)
            )

    def get_pos_embedding(self, vision_input, all_vision_tokens):
        input_shape = vision_input.shape
        pos_embed = VisionTransformer.get_pos_embedding(
            all_vision_tokens.size(1) - self.num_cls_tokens,
            pos_embed=self.pos_embed,
            patches_layout=self.patches_layout,
            input_shape=input_shape,
            first_patch_idx=self.num_cls_tokens,
        )
        return pos_embed


class RGBDTPreprocessor(VerboseNNModule):
    def __init__(
        self,
        rgbt_stem: PatchEmbedGeneric,
        depth_stem: PatchEmbedGeneric,
        img_size: List = (3, 224, 224),
        num_cls_tokens: int = 1,
        pos_embed_fn: Callable = None,
        use_type_embed: bool = False,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()
        stem = rgbt_stem if rgbt_stem is not None else depth_stem
        (
            self.patches_layout,
            self.num_patches,
            self.embed_dim,
        ) = stem.get_patch_layout(img_size)
        self.rgbt_stem = rgbt_stem
        self.depth_stem = depth_stem
        self.use_pos_embed = pos_embed_fn is not None
        self.use_type_embed = use_type_embed
        self.num_cls_tokens = num_cls_tokens

        if self.use_pos_embed:
            self.pos_embedding_helper = pos_embed_fn(
                patches_layout=self.patches_layout,
                num_cls_tokens=num_cls_tokens,
                num_patches=self.num_patches,
                embed_dim=self.embed_dim,
            )
        if self.num_cls_tokens > 0:
            self.cls_token = nn.Parameter(
                torch.zeros(1, self.num_cls_tokens, self.embed_dim)
            )
        if self.use_type_embed:
            self.type_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.init_parameters(init_param_style)

    @torch.no_grad()
    def init_parameters(self, init_param_style):
        if init_param_style == "openclip":
            # OpenCLIP style initialization
            scale = self.embed_dim**-0.5
            if self.use_pos_embed:
                nn.init.normal_(self.pos_embedding_helper.pos_embed)
                self.pos_embedding_helper.pos_embed *= scale

            if self.num_cls_tokens > 0:
                nn.init.normal_(self.cls_token)
                self.cls_token *= scale
        elif init_param_style == "vit":
            self.cls_token.data.fill_(0)
        else:
            raise ValueError(f"Unknown init {init_param_style}")

        if self.use_type_embed:
            nn.init.normal_(self.type_embed)

    def tokenize_input_and_cls_pos(self, input, stem, mask):
        # tokens is of shape B x L x D
        tokens = stem(input)
        assert tokens.ndim == 3
        assert tokens.shape[2] == self.embed_dim
        B = tokens.shape[0]
        if self.num_cls_tokens > 0:
            class_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole class_tokens impl from Phil Wang, thanks
            tokens = torch.cat((class_tokens, tokens), dim=1)
        if self.use_pos_embed:
            pos_embed = self.pos_embedding_helper.get_pos_embedding(input, tokens)
            tokens = tokens + pos_embed
        if self.use_type_embed:
            tokens = tokens + self.type_embed.expand(B, -1, -1)
        return tokens

    def forward(self, vision=None, depth=None, patch_mask=None):
        if patch_mask is not None:
            raise NotImplementedError()

        if vision is not None:
            vision_tokens = self.tokenize_input_and_cls_pos(
                vision, self.rgbt_stem, patch_mask
            )

        if depth is not None:
            depth_tokens = self.tokenize_input_and_cls_pos(
                depth, self.depth_stem, patch_mask
            )

        # aggregate tokens
        if vision is not None and depth is not None:
            final_tokens = vision_tokens + depth_tokens
        else:
            final_tokens = vision_tokens if vision is not None else depth_tokens
        return_dict = {
            "trunk": {
                "tokens": final_tokens,
            },
            "head": {},
        }
        return return_dict


class AudioPreprocessor(RGBDTPreprocessor):
    def __init__(self, audio_stem: PatchEmbedGeneric, **kwargs) -> None:
        super().__init__(rgbt_stem=audio_stem, depth_stem=None, **kwargs)

    def forward(self, audio=None):
        return super().forward(vision=audio)


def build_causal_attention_mask(context_length):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(context_length, context_length, requires_grad=False)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask


class TextPreprocessor(VerboseNNModule):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        causal_masking: bool,
        supply_seq_len_to_head: bool = True,
        num_cls_tokens: int = 0,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.context_length + num_cls_tokens, embed_dim)
        )
        self.causal_masking = causal_masking
        if self.causal_masking:
            mask = build_causal_attention_mask(self.context_length)
            # register the mask as a buffer so it can be moved to the right device
            self.register_buffer("mask", mask)

        self.supply_seq_len_to_head = supply_seq_len_to_head
        self.num_cls_tokens = num_cls_tokens
        self.embed_dim = embed_dim
        if num_cls_tokens > 0:
            assert self.causal_masking is False, "Masking + CLS token isn't implemented"
            self.cls_token = nn.Parameter(
                torch.zeros(1, self.num_cls_tokens, embed_dim)
            )

        self.init_parameters(init_param_style)

    @torch.no_grad()
    def init_parameters(self, init_param_style="openclip"):
        # OpenCLIP style initialization
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

        if init_param_style == "openclip":
            # OpenCLIP style initialization
            scale = self.embed_dim**-0.5
            if self.num_cls_tokens > 0:
                nn.init.normal_(self.cls_token)
                self.cls_token *= scale
        elif init_param_style == "vit":
            self.cls_token.data.fill_(0)
        else:
            raise ValueError(f"Unknown init {init_param_style}")

    def forward(self, text):
        # text tokens are of shape B x L x D
        text_tokens = self.token_embedding(text)
        # concat CLS tokens if any
        if self.num_cls_tokens > 0:
            B = text_tokens.shape[0]
            class_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole class_tokens impl from Phil Wang, thanks
            text_tokens = torch.cat((class_tokens, text_tokens), dim=1)
        text_tokens = text_tokens + self.pos_embed
        return_dict = {
            "trunk": {
                "tokens": text_tokens,
            },
            "head": {},
        }
        # Compute sequence length after adding CLS tokens
        if self.supply_seq_len_to_head:
            text_lengths = text.argmax(dim=-1)
            return_dict["head"] = {
                "seq_len": text_lengths,
            }
        if self.causal_masking:
            return_dict["trunk"].update({"attn_mask": self.mask})
        return return_dict
