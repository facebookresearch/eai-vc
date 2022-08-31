from functools import partial
from typing import Callable, Optional, List

import gym
import torch

from imitation_learning.utils.envs.registry import full_env_registry
from rl_utils.envs.wrappers import TimeLimitMask
from torchrl.envs import GymWrapper


def make_single_gym_env(
    num_envs,
    env_name,
    seed,
    device,
    set_env_settings,
    info_dict_reader=None,
    info_keys=[],
):
    # gym_env = create_env(num_envs, env_name, seed,
    #                     device=device, info_keys=info_keys, **set_env_settings)
    gym_env = GymWrapper(
        create_env(
            num_envs,
            env_name,
            seed,
            device=device,
            info_keys=info_keys,
            **set_env_settings,
        )
    )
    if info_dict_reader is not None:
        gym_env.set_info_dict_reader(info_dict_reader=info_dict_reader)
    return gym_env


def create_env(
    num_envs: int,
    env_id: str,
    seed: int = 0,
    *,
    device: Optional[torch.device] = None,
    context_mode: str = "spawn",
    create_env_fn: Optional[Callable[[int], None]] = None,
    force_multi_proc: bool = False,
    num_frame_stack: Optional[int] = None,
    **kwargs,
):
    found_full_env_cls = full_env_registry.search_env(env_id)
    if found_full_env_cls is not None:
        # print(f"Found {found_full_env_cls} for env {env_id}")
        return found_full_env_cls(num_envs=num_envs, seed=seed, device=device, **kwargs)

    def full_create_env(rank):
        full_seed = seed + rank
        if create_env_fn is None:
            env = gym.make(env_id)
        else:
            env = create_env_fn(full_seed)
        if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimitMask(env)
        env.seed(full_seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(full_seed)
        return env

    envs = partial(full_create_env, rank=0)

    return envs
