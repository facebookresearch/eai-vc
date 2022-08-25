from functools import partial
from typing import Callable, Optional

import gym
import torch

from rl_utils.envs.registry import full_env_registry
from rl_utils.envs.vec_env.dummy_vec_env import DummyVecEnv
from rl_utils.envs.vec_env.shmem_vec_env import ShmemVecEnv
from rl_utils.envs.vec_env.vec_env import VecEnv
from rl_utils.envs.vec_env.vec_monitor import VecMonitor
from rl_utils.envs.wrappers import TimeLimitMask, VecPyTorch, VecPyTorchFrameStack


def create_vectorized_envs(
    env_id: str,
    num_envs: int,
    seed: int = 0,
    *,
    device: Optional[torch.device] = None,
    context_mode: str = "spawn",
    create_env_fn: Optional[Callable[[int], None]] = None,
    force_multi_proc: bool = False,
    num_frame_stack: Optional[int] = None,
    **kwargs,
) -> VecEnv:
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

    envs = [partial(full_create_env, rank=i) for i in range(num_envs)]

    if num_envs > 1 or force_multi_proc:
        envs = ShmemVecEnv(envs, context=context_mode)
    else:
        envs = DummyVecEnv(envs)

    if device is None:
        device = torch.device("cpu")

    envs = VecMonitor(envs)
    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        return VecPyTorchFrameStack(envs, num_frame_stack, device)
    return envs
