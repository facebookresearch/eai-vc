#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from hydra import initialize, compose
from omegaconf import OmegaConf

import habitat

import torch

from vc_models import vc_model_zoo
from habitat.config.default import Config as CN
from habitat_vc.visual_encoder import VisualEncoder


@pytest.fixture(params=vc_model_zoo)
def backbone_config(request, nocluster):
    model_name = request.param

    # Skip everything except randomly-initialized ResNet50 if
    # option "--nocluster" is applied

    if nocluster and "rand" not in model_name:
        pytest.skip()

    with initialize(version_base=None, config_path="../configs/model/transform"):
        transform_cfg = compose(config_name="jitter_and_shift")

    with initialize(
        version_base=None, config_path="../../../vc_models/src/vc_models/conf/model"
    ):
        cfg = compose(
            config_name=model_name,
        )
        cfg.transform = transform_cfg
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = CN(cfg)
        return cfg


def test_env_embedding(backbone_config):
    encoder = VisualEncoder(backbone_config)
    image = torch.zeros((32, 128, 128, 3))

    image = (
        image.permute(0, 3, 1, 2).float() / 255
    )  # convert channels-last to channels-first
    image = encoder.visual_transform(image, 1)

    embedding = encoder(image)

    assert 2 == len(embedding.shape)
    assert embedding.shape[0] == image.shape[0]
