from .cube_env import SimCubeEnv
from .reach_env import ReachEnv
from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

register(
    id="SimCubeEnv-v0",
    entry_point="envs:SimCubeEnv",
    max_episode_steps=1000,
)

register(
    id="ReachEnv-v0",
    entry_point="envs:ReachEnv",
    max_episode_steps=1000,
)
