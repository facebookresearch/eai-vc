from typing import List

import torch
from iopath.common.file_io import g_pathmgr


def load_uru_official_checkpoint(
    checkpoint_paths: List[str],
    map_location: str = "cpu",
):
    """
    Helper function to load one of the official URU checkpoint of:
    https://github.com/facebookresearch/SWAG
    """

    selected_path = None
    for path in checkpoint_paths:
        if g_pathmgr.exists(path):
            selected_path = path
            break

    if selected_path is None:
        raise ValueError(f"No checkpoint found among: {checkpoint_paths}")

    with g_pathmgr.open(selected_path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    state_dict = {}
    for k, v in checkpoint.items():
        k = k.replace("encoder.layers.layer_", "blocks.")
        k = k.replace("self_attention.in_proj_weight", "attn.qkv.weight")
        k = k.replace("self_attention.in_proj_bias", "attn.qkv.bias")
        k = k.replace("ln_", "norm")
        k = k.replace("conv_proj", "patch_embed.proj")
        k = k.replace("self_attention.out_proj", "attn.proj")
        k = k.replace("mlp.linear_", "mlp.fc")
        k = k.replace("class_token", "cls_token")
        k = k.replace("encoder.pos_embedding", "pos_embed")
        k = k.replace("encoder.ln", "norm")
        if k == "pos_embed":
            v = torch.transpose(v, 0, 1)
        state_dict[k] = v
    return state_dict


def expand_pos_embedding(
    state_dict, pos_embedding_key, new_pos_embed_shape, new_pos_embed_init_fn
):
    """
    Expands position embedding of size "k" to a size > "k"
    1. Create a new random position embedding of size > "k"
    2. Copy the old position embedding to the first "k" positions
    """
    old_pos_embed = state_dict[pos_embedding_key]
    # initialize new pos embedding
    new_pos_embed = torch.zeros(list(new_pos_embed_shape))
    new_pos_embed_init_fn(new_pos_embed)
    assert new_pos_embed.ndim == 3
    assert old_pos_embed.ndim == 3
    assert new_pos_embed.shape[0] == old_pos_embed.shape[0]
    assert new_pos_embed.shape[1] > old_pos_embed.shape[1]
    assert new_pos_embed.shape[2] == old_pos_embed.shape[2]
    old_n_tokens = old_pos_embed.shape[1]
    new_pos_embed[:, :old_n_tokens, :] = old_pos_embed
    state_dict[pos_embedding_key] = new_pos_embed
    return state_dict
