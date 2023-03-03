#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
from mjrl.utils.gym_env import GymEnv
from mjrl.utils import tensor_utils
from tqdm import tqdm

logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time as timer
import gc


def toggle_tqdm(rng, debug):
    if debug:
        return tqdm(rng)
    else:
        return rng


def rollout_from_init_states(
    init_states,
    env,
    policy,
    eval_mode=False,
    horizon=1e6,
    debug=False,
) -> list:
    assert isinstance(env, GymEnv)
    assert isinstance(init_states, list)

    num_traj = len(init_states)
    horizon = min(horizon, env.horizon)

    paths = []
    for ep in toggle_tqdm(range(num_traj), debug):
        # set initial state
        env.reset()
        init_state = init_states[ep]
        if env.env_id.startswith("dmc"):
            # dm-control physics backend
            env.env.unwrapped._env.physics.set_state(init_state.astype(np.float64))
            env.env.unwrapped._env.physics.forward()
        else:
            # mujoco_py backend
            env.env.unwrapped.set_env_state(init_state)

        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []

        done = False
        t = 0
        o = env.get_obs()

        while t < horizon and done != True:
            a, agent_info = policy.get_action(o)
            if eval_mode:
                a = agent_info["evaluation"]
            env_info_base = env.get_env_infos()
            next_o, r, done, env_info_step = env.step(a)
            # below is important to ensure correct env_infos for the timestep
            env_info = env_info_step if env_info_base == {} else env_info_base
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o.copy()
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done,
        )
        paths.append(path)

    del env
    gc.collect()
    return paths


if __name__ == "__main__":
    import pickle
    from mjrl.policies.gaussian_mlp import MLP, BatchNormMLP
    from gym_wrapper import env_constructor

    # DMC test
    data_paths = pickle.load(
        open(
            "/checkpoint/maksymets/vc/datasets/dmc-expert-v0.1/dmc_reacher_easy-v1.pickle",
            "rb",
        )
    )
    e = env_constructor(
        env_name="dmc_reacher_easy-v1",
        camera=0,
        embedding_name="r3m_resnet50_ego4d",
        history_window=3,
        seed=12345,
    )
    policy = BatchNormMLP(e.spec, dropout=0.0)
    init_states = [
        p["env_infos"]["internal_state"][0].astype(np.float64) for p in data_paths
    ]
    del data_paths
    gc.collect()

    paths = rollout_from_init_states(
        init_states=init_states,
        env=e,
        policy=policy,
        eval_mode=True,
        horizon=10,  # short horizon for debugging
        debug=True,  # will toggle tqdm
    )

    # Adroit test
    data_paths = pickle.load(
        open(
            "/checkpoint/maksymets/vc/datasets/adroit-expert-v0.1/pen-v0.pickle", "rb"
        )
    )
    e = env_constructor(
        env_name="pen-v0",
        camera=0,
        embedding_name="r3m_resnet50_ego4d",
        history_window=3,
        seed=12345,
    )
    policy = BatchNormMLP(e.spec, dropout=0.0)
    init_states = [p["init_state_dict"] for p in data_paths]
    del data_paths
    gc.collect()

    paths = rollout_from_init_states(
        init_states=init_states,
        env=e,
        policy=policy,
        eval_mode=True,
        horizon=10,  # short horizon for debugging
        debug=True,  # will toggle tqdm
    )

    # Metaworld
    # Current dataset did not store the full state information.
    # So excat scene configuration cannot be recreated.
    # Fixing this requires recollecting the dataset or using the same seed as collection (123)
