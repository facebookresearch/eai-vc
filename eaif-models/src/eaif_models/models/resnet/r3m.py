import os
import omegaconf
import hydra
import copy

import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity

import torchvision
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


VALID_ARGS = [
    "_target_",
    "device",
    "lr",
    "hidden_dim",
    "size",
    "l2weight",
    "l1weight",
    "langweight",
    "tcnweight",
    "l2dist",
    "bs",
]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

R3M_def_path = "eaif_models.models.resnet.r3m.R3M"


def cleanup_config(cfg):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    config.agent["_target_"] = R3M_def_path
    config["device"] = device

    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    config.agent["langweight"] = 0
    return config.agent


def remove_language_head(state_dict):
    keys = state_dict.keys()
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict


def load_r3m(modelid, home):
    if modelid == "resnet50":
        foldername = "r3m_50"
    elif modelid == "resnet34":
        foldername = "r3m_34"
    elif modelid == "resnet18":
        foldername = "r3m_18"
    else:
        raise NameError("Invalid Model ID")

    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    assert os.path.exists(
        os.path.join(home, foldername)
    ), f"Must download R3M models into {home}/{foldername}"

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    r3m_state_dict = remove_language_head(
        torch.load(modelpath, map_location=torch.device(device))["r3m"]
    )
    rep.load_state_dict(r3m_state_dict)
    return rep


epsilon = 1e-8


def do_nothing(x):
    return x


class R3M(nn.Module):
    def __init__(
        self,
        device,
        lr,
        hidden_dim,
        size=34,
        l2weight=1.0,
        l1weight=1.0,
        langweight=1.0,
        tcnweight=0.0,
        l2dist=True,
        bs=16,
    ):
        super().__init__()

        self.device = device
        self.use_tb = False
        self.l2weight = l2weight
        self.l1weight = l1weight
        self.tcnweight = tcnweight  ## Weight on TCN loss (states closer in same clip closer in embedding)
        self.l2dist = l2dist  ## Use -l2 or cosine sim
        self.langweight = langweight  ## Weight on language reward
        self.size = size  ## Size ResNet or ViT
        self.num_negatives = 3

        ## Distances and Metrics
        self.cs = torch.nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = Sigmoid()

        params = []
        ######################################################################## Sub Modules
        ## Visual Encoder
        if size == 18:
            self.outdim = 512
            self.convnet = torchvision.models.resnet18(pretrained=False)
        elif size == 34:
            self.outdim = 512
            self.convnet = torchvision.models.resnet34(pretrained=False)
        elif size == 50:
            self.outdim = 2048
            self.convnet = torchvision.models.resnet50(pretrained=False)

        if self.size == 0:
            self.normlayer = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            self.normlayer = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        self.convnet.fc = Identity()
        self.convnet.train()
        params += list(self.convnet.parameters())
        ## Optimizer
        self.encoder_opt = torch.optim.Adam(params, lr=lr)

    ## Forward Call (im --> representation)
    def forward(self, obs, obs_shape=[3, 224, 224]):
        if obs_shape != [3, 224, 224]:
            preprocess = nn.Sequential(
                T.Resize(256),
                T.CenterCrop(224),
                self.normlayer,
            )
        else:
            preprocess = nn.Sequential(
                self.normlayer,
            )

        ## Input must be [0, 255], [3,244,244]
        obs = obs.float() / 255.0
        obs_p = preprocess(obs)
        h = self.convnet(obs_p)
        return h

    def sim(self, tensor1, tensor2):
        if self.l2dist:
            d = -torch.linalg.norm(tensor1 - tensor2, dim=-1)
        else:
            d = self.cs(tensor1, tensor2)
        return d


_r3m_transforms = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),  # this divides by 255
        T.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 255, 1 / 255, 1 / 255]
        ),  # this will scale bact to [0-255]
    ]
)


def load_model(home_dir, resnet_name="resnet50", metadata=None):
    model = load_r3m(resnet_name, home_dir)
    model = model.module.eval()
    embedding_dim = 2048
    transforms = _r3m_transforms

    return model, embedding_dim, transforms, metadata
