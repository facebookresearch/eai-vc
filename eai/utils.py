import os

import torch

from eai.models.resnet_gn import ResNet
from eai.models.vit import VisionTransformer
from timm.models.vision_transformer import resize_pos_embed


def load_encoder(encoder, path):
    assert os.path.exists(path)
    if isinstance(encoder.backbone, ResNet):
        state_dict = torch.load(path, map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return encoder.load_state_dict(state_dict=state_dict, strict=False)
    elif isinstance(encoder.backbone, VisionTransformer):
        model = encoder.backbone
        state_dict = torch.load(path, map_location="cpu")["model"]
        if state_dict["pos_embed"].shape != model.pos_embed.shape:
            state_dict["pos_embed"] = resize_pos_embed(
                state_dict["pos_embed"],
                model.pos_embed,
                getattr(model, "num_tokens", 1),
                model.patch_embed.grid_size,
            )
        return model.load_state_dict(state_dict=state_dict, strict=False)
    else:
        raise ValueError(f"unknown encoder backbone")
