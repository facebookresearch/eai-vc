from .cube_env import SimCubeEnv
from .reach_env import ReachEnv
from .new_goal_reach_env import NewGoalReachEnv
from .reach_one_goal_env import ReachOneGoalEnv
from .new_action_reach_env import NewActionReachEnv
from .new_reach import NewReachEnv
from .cube_reach import CubeReachEnv
from .gym_cube_env import MoveCubeEnv

from .trifinger_reach_orig import TriFingerReach
from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

register(
    id="SimCubeEnv-v0",
    entry_point="trifinger_envs:SimCubeEnv",
    max_episode_steps=1000,
)

register(
    id="MoveCube-v0",
    entry_point="trifinger_envs:MoveCubeEnv",
    max_episode_steps=1000,
)

register(
    id="ReachEnv-v0",
    entry_point="trifinger_envs:ReachEnv",
    max_episode_steps=2500,
)


register(
    id="NewGoalReachEnv-v0",
    entry_point="trifinger_envs.new_goal_reach_env:NewGoalReachEnv",
    max_episode_steps=1000,
)

register(
    id="CubeReach-v0",
    entry_point="trifinger_envs.cube_reach:CubeReachEnv",
    max_episode_steps=1000,
)

register(
    id="ReachOneGoalEnv-v0",
    entry_point="trifinger_envs:ReachOneGoalEnv",
    max_episode_steps=1000,
)
register(
    id="TrifingerReachOrigEnv-v0",
    entry_point="trifinger_envs:TriFingerReach",
    max_episode_steps=1000,
)
