import warnings

warnings.filterwarnings("ignore")
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
import torch
import numpy as np
from pathlib import Path
from termcolor import colored
from logger import make_dir
from tqdm import tqdm
import torch.nn as nn
import torchvision
import hydra
import kornia.augmentation as K
from kornia.constants import Resample
from einops.layers.torch import Reduce

torch.backends.cudnn.benchmark = True


__ENCODER__ = None
__PREPROCESS__ = None


def make_encoder(cfg):
    """Make an encoder."""
    assert torch.cuda.is_available(), "CUDA is not available"
    features = cfg.get("features", cfg.encoder)
    assert features is not None, "Features must be specified"
    if "mae" in features:
        import mvp

        encoder = mvp.load("vits-mae-hoi").cuda()
        encoder.freeze()
    else:
        arch = "resnet" + str(18 if features.endswith("18") else 50)
        encoder = torchvision.models.__dict__[arch](pretrained=False).cuda()
        if "moco" in features:
            # resnet50
            if features == "moco":
                fn = "moco_v2_800ep_pretrain.pth.tar"
            elif features == "mocodmcontrol":
                fn = "moco_v2_100ep_dmcontrol.pt"
            elif features == "mocometaworld":
                fn = "moco_v2_33ep_metaworld.pt"
            elif features == "mocoego":
                fn = "moco_v2_15ep_pretrain_ego4d.pth.tar"
            elif features == "mocoegodmcontrol":
                fn = "moco_v2_15ep_pretrain_ego_dmcontrol_finetune.pth.tar"
            # resnet18
            elif features == "mocoego18":
                fn = "moco_v2_20ep_pretrain_ego4d_resnet18.pt"
            else:
                raise ValueError("Unknown MOCO model")
            print(colored("Loading MOCO pretrained model: {}".format(fn), "green"))
            state_dict = torch.load(os.path.join(cfg.encoder_dir, fn))["state_dict"]
            state_dict = {
                k.replace("module.encoder_q.", ""): v
                for k, v in state_dict.items()
                if not k.startswith("fc.") and "encoder_k" not in k
            }
            missing_keys, unexpected_keys = encoder.load_state_dict(
                state_dict, strict=False
            )
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        if cfg.get("feature_dims", None) is not None:
            # overwrite forward pass to use earlier features
            # downsample channels with group-wise max pooling (if applicable)
            pool = Reduce("b g c h w -> b 1 c h w", cfg.pool_fn)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                    x = layer(x)
                    if x.shape[-1] == cfg.feature_dims[-1]:
                        break
                if x.shape[1] > cfg.feature_dims[0]:
                    assert (
                        x.shape[1] % cfg.feature_dims[0] == 0
                    ), "Expected number of channels to be divisible by {}".format(
                        cfg.feature_dims[0]
                    )
                    G = x.shape[1] // cfg.feature_dims[0]
                    x = x.view(
                        x.shape[0], G, cfg.feature_dims[0], x.shape[2], x.shape[3]
                    )
                    x = pool(x).squeeze(1)
                    assert (
                        x.shape[1:] == cfg.feature_dims
                    ), "Expected feature dimensions {} but got {}".format(
                        cfg.feature_dims, x.shape[1:]
                    )
                return x

            encoder.forward = lambda x: forward(encoder, x)
            _x = torch.randn(1, 3, 224, 224).cuda()
            print("Pretrained encoder output:", encoder(_x).shape[1:])
        encoder.fc = nn.Identity()
        encoder.eval()
    preprocess = nn.Sequential(
        K.Resize((224, 224), resample=Resample.BICUBIC),
        K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ).cuda()
    return encoder, preprocess


def encode_mae(obs, cfg):
    """Encode one or more observations using a MAE ViT-S model."""
    global __ENCODER__, __PREPROCESS__
    if __ENCODER__ is None:
        __ENCODER__, __PREPROCESS__ = make_encoder(cfg)
    assert isinstance(
        obs, (torch.Tensor, np.ndarray)
    ), "Observation must be a tensor or numpy array"
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
    assert obs.ndim >= 3, "Observation must be at least 3D"
    if obs.ndim == 3:
        obs = obs.unsqueeze(0)
    with torch.no_grad():
        features = __ENCODER__(__PREPROCESS__(obs.cuda() / 255.0))
    return features


def encode_resnet(obs, cfg):
    """Encode one or more observations using a ResNet model."""
    global __ENCODER__, __PREPROCESS__
    if __ENCODER__ is None:
        __ENCODER__, __PREPROCESS__ = make_encoder(cfg)
    assert isinstance(
        obs, (torch.Tensor, np.ndarray)
    ), "Observation must be a tensor or numpy array"
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
    assert obs.ndim >= 3, "Observation must be at least 3D"
    if obs.ndim == 3:
        obs = obs.unsqueeze(0)
    if cfg.modality == "map":
        obs = __PREPROCESS__(obs.cuda() / 255.0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = __ENCODER__(obs)
        # save features to disk
        # features = features.cpu().float()[-1]
        # for i, feature in enumerate(features):
        # 	feature = feature.unsqueeze(0).repeat(3, 1, 1).clip(0, 1)
        # 	# upscale feature with nearest neighbor interpolation
        # 	# feature = K.Resize((140, 140))(feature)
        # 	feature = nn.Upsample(scale_factor=16, mode='nearest')(feature.unsqueeze(0)).squeeze(0)
        # 	torchvision.utils.save_image(feature, f'{cfg.logging_dir}/feature_{i}.png')
        # 	if i == 24:
        # 		exit(0)
    else:
        with torch.no_grad():
            features = __ENCODER__(__PREPROCESS__(obs.cuda() / 255.0))
    return features


def encode(obs, cfg):
    """Encode one or more observations using a pretrained model."""
    features2encoder = {"maehoi": encode_mae}
    fn = features2encoder.get(cfg.features, encode_resnet)
    return fn(obs, cfg)


@hydra.main(config_name="default", config_path="config")
def main(cfg: dict):
    """Encoding an image dataset using a pretrained model."""
    from cfg_parse import parse_cfg
    from dataloader import make_dataset

    assert cfg.get("features", None), "Features must be specified"
    cfg.modality = "pixels"
    cfg = parse_cfg(cfg)
    from env import make_env

    _env = make_env(cfg)
    print(colored(f"\nTask: {cfg.task}", "blue", attrs=["bold"]))

    # Load dataset
    dataset = make_dataset(cfg)

    # Encode dataset
    for episode in tqdm(dataset.episodes):

        # Compute features
        features = encode(episode.obs, cfg).cpu()

        # Save features
        feature_dir = make_dir(
            Path(os.path.dirname(episode.filepath)) / "features" / cfg.features
        )
        torch.save(features, feature_dir / os.path.basename(episode.filepath))


if __name__ == "__main__":
    main()
