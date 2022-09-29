import pytest

from hydra import initialize, compose
from omegaconf import OmegaConf
import torch

from eaif_models import eaif_model_zoo
from habitat.config.default import Config as CN
from habitat_eaif.visual_encoder import VisualEncoder


@pytest.fixture(params=eaif_model_zoo)
def backbone_config(request, nocluster):
    model_name = request.param

    # Skip everything except randomly-initialized ResNet50 if
    # option "--nocluster" is applied
    nocluster_models = ["rand_resnet50_none", "rand_vit_base_none"]
    if nocluster and model_name not in nocluster_models:
        pytest.skip()

    with initialize(
        version_base=None, config_path="../../../eaif-models/src/eaif_models/conf/model"
    ):
        cfg = compose(
            config_name=model_name,
            overrides=["transform._target_=eaif_models.transforms.transform_augment"],
        )
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = CN(cfg)
        return cfg


def test_env_embedding(backbone_config):
    encoder = VisualEncoder(backbone_config)
    image = torch.zeros((32, 128, 128, 3))
    embedding = encoder(image, 1)

    assert 4 == len(embedding.shape)
