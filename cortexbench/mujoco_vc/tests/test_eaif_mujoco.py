#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from mujoco_vc.gym_wrapper import env_constructor
from vc_models import vc_model_zoo

# Full Env list for testing
history_window = 3
seed = 123


@pytest.fixture(params=vc_model_zoo)
def embedding_name(request, nocluster):
    model_name = request.param

    # Skip everything except randomly-initialized ResNet50 if
    # option "--nocluster" is applied
    nocluster_models = ["rand_resnet50_none", "rand_vit_base_none"]
    if nocluster and model_name not in nocluster_models:
        pytest.skip()
    return request.param


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    return request.param


@pytest.fixture(params=["dmc_walker_stand-v1", "relocate-v0"])
def env_name(request):
    return request.param


def test_env_embedding(env_name, embedding_name, device):
    e = env_constructor(
        env_name=env_name,
        embedding_name=embedding_name,
        history_window=history_window,
        seed=seed,
        device=device,
    )
    o = e.reset()
    assert o.shape[0] == e.env.embedding_dim * history_window
    o, r, d, ifo = e.step(e.action_space.sample())
    assert o.shape[0] == e.env.embedding_dim * history_window
