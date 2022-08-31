import os.path as osp
from typing import Dict
from functools import partial

import gym
import gym.spaces as spaces
import hydra
import torch
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig
from imitation_learning.utils.envs import create_env
from imitation_learning.utils.logging import Logger

from imitation_learning.utils import flatten_info_dict_reader
from imitation_learning.utils.evaluator import Evaluator
from torchrl.data import TensorDict
from torchrl.envs import ParallelEnv
from torchrl.envs.utils import step_tensordict
from torchrl.envs import GymWrapper, default_info_dict_reader
from torchrl.envs import GymEnv

from envs.cube_env import ActionType
import envs


@hydra.main(config_path="config", config_name="default")
def main(cfg) -> Dict[str, float]:
    # TODO replace rl-utils functionality for setting seed:
    #     set_seed(cfg.seed)
    device = torch.device(cfg.device)

    set_env_settings = {
        k: hydra_instantiate(v) if isinstance(v, DictConfig) else v
        for k, v in cfg.env.env_settings.items()
    }

    def d_create_env(rank):
        env = gym.make(cfg.env.env_name)
        full_seed = rank + cfg.seed
        env.seed(full_seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(full_seed)
        return env

    dummy_env = gym.make(cfg.env.env_name)
    dummy_env.seed(cfg.seed)
    reader = flatten_info_dict_reader(cfg.info_keys)
    print("Making  " + str(cfg.env.env_name))

    envs = ParallelEnv(
        cfg.num_envs,
        lambda: make_single_gym_env(
            cfg.num_envs,
            cfg.env.env_name,
            cfg.seed,
            device,
            set_env_settings,
            reader,
            cfg.info_keys,
        ),
    )

    steps_per_update = cfg.num_steps * cfg.num_envs
    num_updates = int(cfg.num_env_steps) // steps_per_update

    cfg.obs_shape = dummy_env.observation_space[
        "observation"
    ].shape  # TODO obs dict -> array
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
        vid_dir=logger.vid_path,
        updater=updater,
        logger=logger,
        device=device,
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

    for update_i in range(start_update, num_updates):
        is_last_update = update_i == num_updates - 1
        for step_idx in range(num_steps):
            with torch.no_grad():
                policy.act(td)
            for i in range(9):
                logger.collect_info(
                    f"min_action_dim_{i}", td["action"].min(0).values[i]
                )
                logger.collect_info(
                    f"max_action_dim_{i}", td["action"].max(0).values[i]
                )
            envs.step(td)

            storage_td[:, step_idx] = td
            any_env_done = td["done"].any()
            if any_env_done:
                td.set("reset_workers", td["done"])
                envs.reset(tensordict=td)
                td["next_observation"] = td["observation"]
                logger.collect_env_step_info(td, cfg.info_keys)

            td = step_tensordict(td)
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


def make_single_gym_env(
    num_envs,
    env_name,
    seed,
    device,
    set_env_settings,
    info_dict_reader=None,
    info_keys=[],
):
    temp_vis = False
    temp_collision = True
    temp_disable_cam = True
    SIM_TIME_STEP = 0.004

    # TODO pass this into cube env
    cube_kwargs = {
        "env": "SimCubeEnv",
        "goal_pose": None,  # passing None to sample a random trajectory
        "action_type": ActionType.TORQUE,
        "visualization": temp_vis,
        "no_collisions": temp_collision,
        "enable_cameras": (not temp_disable_cam),
        "finger_type": "trifingerpro",
        "time_step": SIM_TIME_STEP,
        "camera_delay_steps": 0,
    }

    cube_env = gym.make(env_name)
    gym_env = GymWrapper(cube_env)
    tensordict = gym_env.reset()
    tensordict = gym_env.rand_step(tensordict)
    return gym_env


if __name__ == "__main__":
    main()
