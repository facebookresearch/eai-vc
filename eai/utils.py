import os

import torch


def load_encoder(encoder, path, state_dict_key="teacher"):
    assert os.path.exists(path)
    state_dict = torch.load(path, map_location="cpu")[state_dict_key]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return encoder.load_state_dict(state_dict=state_dict, strict=False)
