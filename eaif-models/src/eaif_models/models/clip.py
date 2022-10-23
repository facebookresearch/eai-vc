import torch
from timm.models.vision_transformer import resize_pos_embed


# Taken from https://github.com/rwightman/pytorch-image-models/blob/9709dbaaa95ee603841fcc055a96327f8edf4320/timm/models/vision_transformer.py#L709
def _convert_openai_clip(state_dict, model):
    out_dict = {}
    swaps = [
        ("visual.", ""),
        ("conv1", "patch_embed.proj"),
        ("positional_embedding", "pos_embed"),
        ("transformer.resblocks.", "blocks."),
        ("ln_pre", "norm_pre"),
        ("ln_post", "norm"),
        ("ln_", "norm"),
        ("in_proj_", "qkv."),
        ("out_proj", "proj"),
        ("mlp.c_fc", "mlp.fc1"),
        ("mlp.c_proj", "mlp.fc2"),
    ]
    for k, v in state_dict.items():
        if not k.startswith("visual."):
            continue
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == "proj":
            k = "head.weight"
            v = v.transpose(0, 1)
            out_dict["head.bias"] = torch.zeros(v.shape[0])
        elif k == "class_embedding":
            k = "cls_token"
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == "pos_embed":
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                # To resize pos embedding when using model at different size from pretrained weights
                v = resize_pos_embed(
                    v,
                    model.pos_embed,
                    0
                    if getattr(model, "no_embed_class")
                    else getattr(model, "num_prefix_tokens", 1),
                    model.patch_embed.grid_size,
                )
        out_dict[k] = v
    return out_dict


def load_clip_vit_checkpoint(model, checkpoint_path=None):
    if checkpoint_path is None:
        return model

    clip_model = torch.jit.load(checkpoint_path, map_location="cpu")
    state_dict = clip_model.state_dict()
    state_dict = _convert_openai_clip(state_dict, model)

    if model.global_pool:
        # remove layer that start with norm
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("norm")}
        # add fc_norm in the state dict from the model
        state_dict["fc_norm.weight"] = model.fc_norm.weight
        state_dict["fc_norm.bias"] = model.fc_norm.bias

    model.load_state_dict(state_dict)
    return model
