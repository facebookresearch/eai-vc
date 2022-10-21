import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn

try:
    import timm
    from timm.models.layers import Mlp, to_2tuple
    from timm.models.layers.attention_pool2d import (
        AttentionPool2d as AbsAttentionPool2d,
        RotAttentionPool2d,
    )
except ImportError as e:
    timm = None

try:
    import xformers
    from xformers.ops import memory_efficient_attention, unbind
except ImportError:
    memory_efficient_attention = None


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features**-0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def stem(self, x):
        for conv, bn in [
            (self.conv1, self.bn1),
            (self.conv2, self.bn2),
            (self.conv3, self.bn3),
        ]:
            x = self.relu(bn(conv(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = x.to(self.conv1.weight.dtype)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.to(torch.float32))
        return ret.to(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        # NOTE I do not know why this is the default. Slower than nn.GELU or nn.SiLU and use more GPU memory
        return x * torch.sigmoid(1.702 * x)


class XformerAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if memory_efficient_attention is None:
            raise ValueError(
                "To use xformers efficient attention, please install xformers (use an interactive node): pip install git+https://github.com/facebookresearch/xformers.git"
            )
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        x = memory_efficient_attention(
            q, k, v, op=xformers.ops.MemoryEfficientAttentionCutlassFwdFlashBwOp
        )
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        act_layer: Callable = nn.GELU,
        use_xformer: bool = False,
    ):
        super().__init__()

        if use_xformer:
            if memory_efficient_attention is None:
                raise ValueError(
                    "To use xformers efficient attention, please install xformers (use an interactive node): pip install git+https://github.com/facebookresearch/xformers.git"
                )

            self.attn = XformerAttention(d_model, n_head)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", act_layer()),
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
        return self.attn(
            x, key=x, value=x, need_weights=False, attn_mask=self.attn_mask
        )[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        act_layer: Callable = nn.GELU,
        use_xformer: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
                    act_layer=act_layer,
                    use_xformer=use_xformer,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, use_checkpoint=False):
        for blk in self.resblocks:
            if use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


class VisualTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        act_layer: Callable = nn.GELU,
        use_xformer: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((image_size // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads, act_layer=act_layer, use_xformer=use_xformer
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, use_checkpoint=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, use_checkpoint=use_checkpoint)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TimmModel(nn.Module):
    def __init__(
        self,
        model_name,
        embed_dim,
        image_size=224,
        pool="avg",
        proj="linear",
        drop=0.0,
        pretrained=False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please `pip install timm` to use timm models.")

        self.image_size = to_2tuple(image_size)
        self.trunk = timm.create_model(model_name, pretrained=pretrained)
        self.trunk.reset_classifier(0, global_pool="")
        feat_size = self.trunk.default_cfg.get("pool_size", None)
        feature_ndim = 1 if not feat_size else 2
        prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        if feature_ndim == 2:
            assert pool, "pooling layer needed for 2d feature output"
            if pool == "abs_attn":
                assert feature_ndim == 2
                head_layers["pool"] = AbsAttentionPool2d(
                    prev_chs, feat_size=feat_size, out_features=embed_dim
                )
                prev_chs = embed_dim
            elif pool == "rot_attn":
                assert feature_ndim == 2
                head_layers["pool"] = RotAttentionPool2d(
                    prev_chs, out_features=embed_dim
                )
                prev_chs = embed_dim
            elif pool == "avg":
                assert proj, "projection layer needed if avg pooling used"
                head_layers["pool"] = nn.AdaptiveAvgPool2d(1)
        else:
            # NOTE timm transformers will be changed in the future to return unpooled
            # outputs when head is disabled, this is not he case right now and code be needed
            # here for token extraction or pooling
            pass

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if proj == "linear":
            head_layers["drop"] = nn.Dropout(drop)
            head_layers["proj"] = nn.Linear(prev_chs, embed_dim)
        elif proj == "mlp":
            head_layers["mlp"] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=drop)

        self.head = nn.Sequential(head_layers)

    def forward(self, x):
        x = self.trunk(x)
        x = self.head(x)
        return x


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    heads: Optional[int] = None
    width: int = 768
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    use_xformer: bool = False
    timm_model_name: str = (
        None  # a valid model name overrides layers, width, patch_size
    )
    timm_model_pretrained: bool = (
        False  # use (imagenet) pretrained weights for named model
    )
    timm_pool: str = (
        "avg"  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    )
    timm_proj: str = (
        "linear"  # linear projection for timm model output ('linear', 'mlp', '')
    )


@dataclass
class CLIPTextCfg:
    context_length: int
    vocab_size: int
    width: int
    heads: int
    layers: int
    use_xformer: bool = False


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
    ):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        self.context_length = text_cfg.context_length

        # OpenAI models are  pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.
        act_layer = QuickGELU if quick_gelu else nn.GELU

        if vision_cfg.timm_model_name:
            self.visual = TimmModel(
                vision_cfg.timm_model_name,
                pretrained=vision_cfg.timm_model_pretrained,
                pool=vision_cfg.timm_pool,
                proj=vision_cfg.timm_proj,
                embed_dim=embed_dim,
                image_size=vision_cfg.image_size,
            )
            act_layer = (
                nn.GELU
            )  # so that text transformer doesn't use QuickGELU w/ timm models
        elif isinstance(vision_cfg.layers, (tuple, list)):
            vision_heads = vision_cfg.width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width,
            )
        else:
            vision_heads = vision_cfg.width // 64
            self.visual = VisualTransformer(
                image_size=vision_cfg.image_size,
                patch_size=vision_cfg.patch_size,
                width=vision_cfg.width,
                layers=vision_cfg.layers,
                use_xformer=vision_cfg.use_xformer,
                heads=vision_cfg.heads or vision_heads,
                output_dim=embed_dim,
            )

        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            use_xformer=text_cfg.use_xformer,
            attn_mask=self.build_attention_mask(),
            act_layer=act_layer,
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, text_cfg.width)
        )
        self.ln_final = LayerNorm(text_cfg.width)

        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if hasattr(self.visual, "init_parameters"):
            self.visual.init_parameters()

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

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return next(self.visual.parameters()).dtype

    def encode_image(self, image, use_checkpoint=False):

        return self.visual(image, use_checkpoint=use_checkpoint)

    def encode_text(self, text, use_checkpoint=False):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, use_checkpoint=use_checkpoint)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, use_checkpoint=False):
        if image is None:
            return self.encode_text(text, use_checkpoint=use_checkpoint)
        elif text is None:
            return self.encode_image(image, use_checkpoint=use_checkpoint)
        image_features = self.encode_image(image, use_checkpoint=use_checkpoint)
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.encode_text(text, use_checkpoint=use_checkpoint)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features, self.logit_scale.exp()


class CLIPSM(CLIP):
    """
    CLIP with shared modalities. Even the attention masks will be the same
        since the Transformer object is being overwritten.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.transformer
        self.transformer = self.visual.transformer


class CLIPShareAllLayersExceptSomeInResBlock(CLIP):
    """
    CLIP with shared modalities but separate 0 or more layers.
    Also all tensors (eg attention masks in the resblocks) remain separate.
    """

    @staticmethod
    def check_layer_in_list(
        lname: str, layer: nn.Module, layer_list: Tuple[Union[str, nn.Module]]
    ):
        for i in range(len(layer_list)):
            if isinstance(layer_list[i], str):
                if lname == layer_list[i]:
                    return True
            elif isinstance(layer_list[i], nn.Module):
                if isinstance(layer, type(layer_list[i])):
                    return True
        return False

    def __init__(
        self,
        *args,
        separate_layers: Tuple[Union[str, nn.Module]] = (),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for i in range(len(self.visual.transformer.resblocks)):
            # Note that named_children will only find parameters in the layer
            # Tensors and buffers like attn_mask will not be copied and kept separate
            for lname, layer in self.visual.transformer.resblocks[i].named_children():
                if not self.check_layer_in_list(lname, layer, separate_layers):
                    delattr(self.transformer.resblocks[i], lname)
                    setattr(self.transformer.resblocks[i], lname, layer)


class CLIPSMSepLN(CLIPShareAllLayersExceptSomeInResBlock):
    """
    CLIP with shared modalities but separate LN
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, separate_layers=("ln_1", "ln_2"), **kwargs)


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=True,  # OpenAI models were trained with QuickGELU
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)
    return model.eval()


class MultiModalZeroShotEvalWrapperCLIP(nn.Module):
    """
    Takes a multimodal input and computes features for each modality using
    corresponding models.
    TODO this should somehow inherit from the other wrappers....
    """

    def __init__(
        self,
        clip_model,
        label_strings,
        image_output_key="image_embed",
        text_output_key="text_embed",
        logit_scale_output_key="logit_scale",
        clip_model_init_weights=None,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model
        if clip_model_init_weights:
            sd = torch.load(clip_model_init_weights)
            msg = self.clip_model.load_state_dict(sd)
            print(f"Initialized model from {clip_model_init_weights}: {msg}")

        self.label_strings = label_strings
        # To be used for classifying actions by matching to
        # Will be set before each validation run
        self.target_text_features = None
        self.text_output_key = text_output_key
        self.image_output_key = image_output_key
        self.logit_scale_output_key = logit_scale_output_key

    def _get_feat_from_dict(self, feat_dict):
        """
        This function is needed because the 2 trunks in here are MIMOWrappers,
        which return a dict of features. So this function reads out the features.
        Assumes only 1 output feature
        """
        assert len(feat_dict) == 1
        return feat_dict[list(feat_dict.keys())[0]]

    def parse_kwargs_per_trunk(self, kwargs):
        if not kwargs:
            return {}, {}
        assert "vision_trunk" in kwargs, "specify kwargs for vision_trunk"
        assert "text_trunk" in kwargs, "specify kwargs for text trunk"
        return kwargs["vision_trunk"], kwargs["text_trunk"]

    def forward_train(self, batch, *args, **kwargs):
        assert isinstance(batch, Mapping)
        outputs = {}
        kwargs_vision, kwargs_text = self.parse_kwargs_per_trunk(kwargs)
        for key, sub_batch in batch.items():
            outputs[key] = {}
            image_feature, text_feature, logit_scale = self.clip_model(
                image=sub_batch.vision,
                text=sub_batch.text,
                **kwargs_vision,
                **kwargs_text,
            )
            outputs[key].update({self.image_output_key: image_feature})
            outputs[key].update({self.text_output_key: text_feature})
            outputs[key].update({self.logit_scale_output_key: logit_scale})
        return outputs

    def on_validation_epoch_start(self):
        assert isinstance(self.label_strings, Mapping)
        self.target_text_features = {}
        for key, label_strings in self.label_strings.items():
            num_classes = len(label_strings)
            logging.info(
                f"Validation start: Computing target string features for "
                f"{num_classes} classes in {key}..."
            )

            model_device = next(self.clip_model.parameters()).device
            with torch.no_grad():
                all_label_embeddings = []
                num_label_names_per_class = []
                for cls_idx in range(num_classes):
                    num_label_names_per_class.append(len(label_strings[cls_idx]))
                    for templates_per_label_name in label_strings[cls_idx]:
                        templates_per_label_name_feats = self.clip_model(
                            image=None,
                            text=templates_per_label_name.to(model_device),
                        )
                        all_label_embeddings.append(templates_per_label_name_feats)

                target_text_features = torch.stack(all_label_embeddings)
                # normalize all text features
                target_text_features = torch.nn.functional.normalize(
                    target_text_features, dim=-1, p=2
                )

                # mean across templates (dim=1)
                assert target_text_features.ndim == 3
                target_text_features = target_text_features.mean(dim=1)

                # renormalize
                target_text_features = torch.nn.functional.normalize(
                    target_text_features, dim=-1, p=2
                )
            # transpose since we can easily compute an inner_product with image features
            target_text_features = target_text_features.t()
            num_label_names_per_class = np.array(num_label_names_per_class)
            assert target_text_features.shape[1] == num_label_names_per_class.sum()
            num_label_names_per_class_cumsum = np.cumsum(
                num_label_names_per_class
            ).tolist()
            num_label_names_per_class_cumsum.insert(0, 0)
            num_label_names_per_class_cumsum_idx = (
                torch.Tensor(num_label_names_per_class_cumsum).long().to(model_device)
            )
            self.target_text_features[key] = (
                target_text_features,
                num_label_names_per_class_cumsum_idx,
                num_classes,
            )

            logging.info("...computing target strings done.")

    def on_validation_epoch_end(self):
        # Setting back to None so we don't mistakenly use the same features
        # again in the next epoch the evaluation is done. Must be recomputed
        # in on_validation_epoch_start.
        self.target_text_features = None

    def forward_val(self, batch, *args, **kwargs):
        assert isinstance(batch, Mapping)
        outputs = {}
        kwargs_vision, _ = self.parse_kwargs_per_trunk(kwargs)
        for key, sub_batch in batch.items():
            image_feature = self.clip_model(
                image=sub_batch.vision, text=None, *args, **kwargs_vision
            )
            image_feature = torch.nn.functional.normalize(image_feature, p=2, dim=-1)
            (
                target_text_feature,
                num_label_names_per_class_cumsum_idx,
                num_classes,
            ) = self.target_text_features[key]
            img_txt_matches = image_feature @ target_text_feature
            # aggregate predictions per class by computing a max over the logits
            # for label_names in that class
            aggregated_logits = []
            for k in range(num_classes):
                logits_per_label_names = img_txt_matches[
                    :,
                    num_label_names_per_class_cumsum_idx[
                        k
                    ] : num_label_names_per_class_cumsum_idx[k + 1],
                ]
                aggregated_logits.append(logits_per_label_names.max(dim=1).values)
            aggregated_logits = torch.stack(aggregated_logits).t()
            assert aggregated_logits.shape[1] == num_classes
            outputs[key] = aggregated_logits
        return outputs

    def forward(self, batch, *args, **kwargs):
        if self.target_text_features is not None:
            # The text features are set by the on_validation_epoch_start function
            # Hence the validation epoch has started, so that forward should be called.
            # When that epoch finishes, the text features are set back to None
            return self.forward_val(batch, *args, **kwargs)
        return self.forward_train(batch, *args, **kwargs)
