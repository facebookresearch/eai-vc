import pytest
import os
import hydra
import omegaconf
import numpy as np
import torch
import torchvision
import PIL

import eaif_models
from eaif_models.utils import get_model_tag


eaif_models_abs_path = os.path.dirname(os.path.abspath(eaif_models.__file__))


def get_config_path(model_name):
    cfg_path = os.path.join(eaif_models_abs_path, "conf", "model", f"{model_name}")
    if os.path.isdir(cfg_path):
        pytest.skip()
    cfg_path += ".yaml"
    return cfg_path


@pytest.mark.parametrize("model_name", eaif_models.eaif_model_zoo)
def test_cfg_name(model_name):
    cfg_path = get_config_path(model_name)

    model_cfg = omegaconf.OmegaConf.load(cfg_path)
    assert get_model_tag(model_cfg.metadata) == model_name


@pytest.mark.parametrize("model_name", eaif_models.eaif_model_zoo)
def test_model_loading(model_name):
    """
    Test creating the model architecture without loading the checkpoint.
    """
    cfg_path = get_config_path(model_name)

    model_cfg = omegaconf.OmegaConf.load(cfg_path)
    if "model" in model_cfg.model:
        model = hydra.utils.call(model_cfg.model.model)
    else:
        model = hydra.utils.call(model_cfg.model)

    assert model.training
    assert next(model.parameters()).device == torch.device("cpu")

    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224))


@pytest.mark.parametrize("model_name", eaif_models.eaif_model_zoo)
def test_model_loading_with_checkpoint(model_name, nocluster):
    """
    Test creating the model architecture as well as loading the checkpoint.
    """
    if nocluster:
        pytest.skip()

    cfg_path = get_config_path(model_name)

    model_cfg = omegaconf.OmegaConf.load(cfg_path)
    model, embedding_dim, transform, metadata = hydra.utils.call(model_cfg)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(embedding_dim, int)
    assert isinstance(
        transform, (torch.nn.Module, torchvision.transforms.transforms.Compose)
    )
    assert isinstance(metadata, omegaconf.Container)

    assert model.training
    assert next(model.parameters()).device == torch.device("cpu")

    with torch.no_grad():
        # Test transform
        imarray = np.random.rand(100, 100, 3) * 255
        img = PIL.Image.fromarray(imarray.astype("uint8")).convert("RGB")
        transformed_img = transform(img).unsqueeze(0)

        # Test embedding dim is correct
        assert torch.Size([1, embedding_dim]) == model(transformed_img).shape
