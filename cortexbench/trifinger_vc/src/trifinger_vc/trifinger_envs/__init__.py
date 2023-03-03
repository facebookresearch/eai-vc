#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from .cube_reach import CubeReachEnv
from .gym_cube_env import MoveCubeEnv

from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

register(
    id="MoveCube-v0",
    entry_point="trifinger_envs:MoveCubeEnv",
    max_episode_steps=1000,
)


register(
    id="CubeReach-v0",
    entry_point="trifinger_envs.cube_reach:CubeReachEnv",
    max_episode_steps=1000,
)
