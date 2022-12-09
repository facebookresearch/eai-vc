# to run with R3M model and image observation space: ```python run.py logger.wb_entity=snsilwal logger.wb_proj_name=tri_reach env=reach_env pretrained_model="R3M" evaluator=pretrained visual_observation=True````

import os.path as osp
import numpy as np
from typing import Dict

import gym.spaces as spaces
import hydra
import torch
from hydra.utils import instantiate as hydra_instantiate
from imitation_learning.utils.logging import Logger

from imitation_learning.utils.evaluator import Evaluator
from torchrl.data import TensorDict
from torchrl.envs import ParallelEnv

try:
    from torchrl.envs.utils import step_tensordict as step_tensordict
except ImportError:
    from torchrl.envs.utils import step_mdp as step_tensordict
from torchrl.envs.libs.gym import GymWrapper
import trifinger_envs
from rl_utils import common
import random
import gym


def compute_ftip_dists(obs):
    ftip_dists = []
    num_fingers = 3
    for i in range(num_fingers):
        ftip_dists.append(
            torch.norm(
                obs[:, 3 * i : 3 + (3 * i)] - obs[:, 9 + 3 * i : 12 + (3 * i)], dim=1
            )
        )
    return ftip_dists


def get_representation(obs):
    return


def create_trifinger_env(env_name, seed, env_kwargs, sample_radius, sample_goals):
    # TODO fix gym registration
    if env_name == "ReachEnv-v0":
        env = trifinger_envs.reach_env.ReachEnv(**env_kwargs)
    # hydra param env: new_reach
    elif env_name == "ReachEnv-v1":
        # no logic for random starts here
        env_kwargs["sample_radius"] = sample_radius
        env_kwargs["randomize_all"] = sample_goals
        env = trifinger_envs.new_reach.NewReachEnv(**env_kwargs)
    env.seed(seed)
    gym_env = GymWrapper(env)
    tensordict = gym_env.reset()
    tensordict = gym_env.rand_step(tensordict)
    return gym_env


@hydra.main(config_path="config", config_name="default")
def main(cfg) -> Dict[str, float]:
    # TODO replace rl-utils functionality for setting seed:
    device = torch.device(cfg.device)
    torch.use_deterministic_algorithms(True)
    env_kwargs = {"max_goal_dist": cfg.max_goal_dist, "camera_id": cfg.camera_id}
    dummy_env = create_trifinger_env(
        cfg.env.env_name, cfg.seed, env_kwargs, cfg.start_radius, cfg.sample_goals
    )
    print("Making  " + str(cfg.env.env_name))
    envs = ParallelEnv(
        cfg.num_envs,
        lambda: create_trifinger_env(
            cfg.env.env_name, cfg.seed, {}, cfg.start_radius, cfg.sample_goals
        ),
    )
    envs.set_seed(cfg.seed)
    random.seed(cfg.seed)
    common.core_utils.set_seed(cfg.seed)

    steps_per_update = cfg.num_steps * cfg.num_envs
    num_updates = int(cfg.num_env_steps) // steps_per_update

    cfg.obs_shape = dummy_env.observation_space.shape

    cfg.action_dim = dummy_env.action_space.shape[0]
    cfg.action_is_discrete = isinstance(cfg.action_dim, spaces.Discrete)

    cfg.total_num_updates = num_updates

    logger: Logger = hydra_instantiate(cfg.logger, full_cfg=cfg)
    policy = hydra_instantiate(cfg.policy)
    policy = policy.to(device)
    updater = hydra_instantiate(cfg.policy_updater, policy=policy, device=device)
    evaluator: Evaluator = hydra_instantiate(
        cfg.evaluator,
        envs=envs,
        num_render=cfg.num_eval_episodes,
        vid_dir=logger.vid_path,
        updater=updater,
        logger=logger,
        device=device,
        preprocess_func=get_representation,
    )

    start_update = 0
    if cfg.load_checkpoint is not None:
        ckpt = torch.load(cfg.load_checkpoint)
        updater.load_state_dict(ckpt["updater"], should_load_opt=cfg.resume_training)
        if cfg.load_policy:
            policy.load_state_dict(ckpt["policy"])
        if cfg.resume_training:
            start_update = ckpt["update_i"] + 1

    eval_info = {"run_name": logger.run_name}

    if cfg.only_eval:
        eval_result = evaluator.evaluate(policy, cfg.num_eval_episodes, 0)
        logger.collect_infos(eval_result, "eval.", no_rolling_window=True)
        eval_info.update(eval_result)
        logger.interval_log(0, 0)
        logger.close()

        return eval_info

    num_steps = cfg.num_steps
    td = envs.reset()
    storage_td = TensorDict({}, batch_size=[cfg.num_envs, num_steps], device=device)
    max_env_step_idx = 19

    for update_i in range(start_update, num_updates):
        is_last_update = update_i == num_updates - 1

        # num_steps are # for each update, whereas env_steps keep track of
        # how many steps are left for an epsiode in the env
        env_steps = 0
        for step_idx in range(num_steps):
            with torch.no_grad():
                policy.act(td)
            envs.step(td)
            storage_td[:, step_idx] = td
            if env_steps >= max_env_step_idx:
                completed_td = td["observation"].detach()
                ftip_distances = torch.norm(
                    completed_td[:, :9] - completed_td[:, 9:], dim=1
                )

                per_finger_dists = compute_ftip_dists(completed_td)
                avg_ftip_dist = 0
                for f in per_finger_dists:
                    avg_ftip_dist += f
                avg_ftip_dist = avg_ftip_dist / 3
                logger.collect_info(
                    "avg_ftip_dist",
                    avg_ftip_dist.mean().item(),
                    no_rolling_window=True,
                )
                logger.collect_info(
                    "ftip0_dist",
                    per_finger_dists[0].mean().item(),
                    no_rolling_window=True,
                )
                logger.collect_info(
                    "ftip1_dist",
                    per_finger_dists[1].mean().item(),
                    no_rolling_window=True,
                )
                logger.collect_info(
                    "ftip2_dist",
                    per_finger_dists[2].mean().item(),
                    no_rolling_window=True,
                )

                # success_rate = percentage done and w/ distance less than X / all storage_td
                success_rate = (
                    avg_ftip_dist < 0.02 * td["done"].T
                ).sum() / avg_ftip_dist.shape[0]
                logger.collect_info(
                    "success_total_size",
                    ftip_distances.shape[0],
                    no_rolling_window=True,
                )
                logger.collect_info(
                    "success_rate", success_rate, no_rolling_window=True
                )
                logger.collect_info(
                    "fingertip_dist",
                    ftip_distances.mean().item(),
                    no_rolling_window=True,
                )
                # logger.collect_env_step_info(td, cfg.info_keys)
                for k in cfg.info_keys:
                    logger.collect_info(k, td[k].numpy(), no_rolling_window=True)
                td.set("reset_workers", td["done"])
                envs.reset(tensordict=td)
                td["next_observation"] = td["observation"]
                env_steps = -1

            td = step_tensordict(td)
            env_steps += 1

        logger.collect_info(
            "cumulative_reward",
            storage_td["reward"].mean().item(),
            no_rolling_window=True,
        )
        logger.collect_info(
            "last_reward",
            storage_td["reward"][:, -1].mean().item(),
            no_rolling_window=True,
        )

        updater.update(policy, storage_td, logger, envs=envs)

        if cfg.eval_interval != -1 and (
            update_i % cfg.eval_interval == 0 or is_last_update
        ):
            with torch.no_grad():
                eval_result = evaluator.evaluate(
                    policy, cfg.num_eval_episodes, update_i
                )
            logger.collect_infos(eval_result, "eval.", no_rolling_window=True)
            eval_info.update(eval_result)

        if cfg.log_interval != -1 and (
            update_i % cfg.log_interval == 0 or is_last_update
        ):
            logger.interval_log((update_i + 1), steps_per_update * (update_i + 1))

        if cfg.save_interval != -1 and (
            (update_i + 1) % cfg.save_interval == 0 or is_last_update
        ):
            save_name = osp.join(logger.save_path, f"ckpt.{update_i}.pth")
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "updater": updater.state_dict(),
                    "update_i": update_i,
                },
                save_name,
            )
            print(f"Saved to {save_name}")
            eval_info["last_ckpt"] = save_name

    logger.close()
    return eval_info


if __name__ == "__main__":
    main()
