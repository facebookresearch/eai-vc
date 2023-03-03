#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import os
import hydra
import omegaconf
import numpy as np
import torch
import torchvision
import PIL

import vc_models
from vc_models.utils import get_model_tag
from vc_models.models.vit import model_utils as vit_model_utils


vc_models_abs_path = os.path.dirname(os.path.abspath(vc_models.__file__))


def get_config_path(model_name):
    cfg_path = os.path.join(vc_models_abs_path, "conf", "model", f"{model_name}")
    if os.path.isdir(cfg_path):
        pytest.skip()
    cfg_path += ".yaml"
    return cfg_path


@pytest.mark.parametrize("model_name", vc_models.vc_model_zoo)
def test_cfg_name(model_name):
    cfg_path = get_config_path(model_name)

    model_cfg = omegaconf.OmegaConf.load(cfg_path)
    model_tag = get_model_tag(model_cfg.metadata)

    if model_name == vit_model_utils.VC1_LARGE_NAME:
        assert model_tag == 'mae_vit_large_patch16_ego_imagenet_inav_182_epochs'
    elif model_name == vit_model_utils.VC1_BASE_NAME:
        assert model_tag == 'mae_vit_base_patch16_ego_imagenet_inav_182_epochs'
    else:
        assert model_tag == model_name


@pytest.mark.parametrize("model_name", vc_models.vc_model_zoo)
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


@pytest.mark.parametrize("model_name", vc_models.vc_model_zoo)
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
