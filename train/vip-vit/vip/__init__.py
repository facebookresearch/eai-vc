# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from vip.models.model_vip import VIP

import os
from os.path import expanduser
import omegaconf
import hydra
import torch
from torch.hub import load_state_dict_from_url
import copy

VALID_ARGS = [
    "_target_",
    "device",
    "lr",
    "hidden_dim",
    "size",
    "l2weight",
    "l1weight",
    "num_negatives",
]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def cleanup_config(cfg):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    config.agent["_target_"] = "vip.VIP"
    config["device"] = device

    return config.agent


def load_vip(modelid="resnet50"):
    home = os.path.join(expanduser("~"), ".vip")

    if not os.path.exists(os.path.join(home, modelid)):
        os.makedirs(os.path.join(home, modelid))
    folderpath = os.path.join(home, modelid)
    modelpath = os.path.join(home, modelid, "model.pt")
    configpath = os.path.join(home, modelid, "config.yaml")

    # Default download from PyTorch S3 bucket; use G-Drive as a backup.
    try:
        if modelid == "resnet50":
            modelurl = "https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt"
            configurl = "https://pytorch.s3.amazonaws.com/models/rl/vip/config.yaml"
        else:
            raise NameError("Invalid Model ID")
        if not os.path.exists(modelpath):
            load_state_dict_from_url(modelurl, folderpath)
            load_state_dict_from_url(configurl, folderpath)
    except:
        if modelid == "resnet50":
            modelurl = (
                "https://drive.google.com/uc?id=1LuCFIV44xTZ0GLmLwk36BRsr9KjCW_yj"
            )
            configurl = (
                "https://drive.google.com/uc?id=1XSQE0gYm-djgueo8vwcNgAiYjwS43EG-"
            )
        else:
            raise NameError("Invalid Model ID")
        if not os.path.exists(modelpath):
            import gdown

            gdown.download(modelurl, modelpath, quiet=False)
            gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    vip_state_dict = torch.load(modelpath, map_location=torch.device(device))["vip"]
    rep.load_state_dict(vip_state_dict)
    return rep
