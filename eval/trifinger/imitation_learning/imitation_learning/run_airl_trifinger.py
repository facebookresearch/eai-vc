import os
import os.path as osp
from collections import defaultdict
from typing import Dict, Optional

import gym.spaces as spaces
import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf
from rl_helper.common import Evaluator, compress_dict, get_size_for_space, set_seed
from rl_helper.envs import create_vectorized_envs
from rl_helper.logging import Logger

from imitation_learning.policy_opt.policy import Policy
from imitation_learning.policy_opt.ppo import PPO
from imitation_learning.policy_opt.storage import RolloutStorage
from imitation_learning.run_mirl_trifinger import TrifingerEvaluator


@hydra.main(config_path="config/airl", config_name="trifinger")
def main(cfg) -> Dict[str, float]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    set_env_settings = {
        k: hydra_instantiate(v) if isinstance(v, DictConfig) else v
        for k, v in cfg.env.env_settings.items()
    }
    envs = create_vectorized_envs(
        cfg.env.env_name,
        cfg.num_envs,
        seed=cfg.seed,
        device=device,
        **set_env_settings,
    )

    steps_per_update = cfg.num_steps * cfg.num_envs
    num_updates = int(cfg.num_env_steps) // steps_per_update

    cfg.obs_shape = envs.observation_space.shape
    cfg.action_dim = get_size_for_space(envs.action_space)
    cfg.action_is_discrete = isinstance(cfg.action_dim, spaces.Discrete)
    cfg.total_num_updates = num_updates

    logger: Logger = hydra_instantiate(cfg.logger, full_cfg=cfg)

    storage: RolloutStorage = hydra_instantiate(cfg.storage, device=device)
    policy: Policy = hydra_instantiate(cfg.policy)
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

    obs = envs.reset()
    storage.init_storage(obs)

    for update_i in range(start_update, num_updates):
        is_last_update = update_i == num_updates - 1
        for step_idx in range(cfg.num_steps):
            with torch.no_grad():
                act_data = policy.act(
                    storage.get_obs(step_idx),
                    storage.recurrent_hidden_states[step_idx],
                    storage.masks[step_idx],
                )
            next_obs, reward, done, info = envs.step(act_data["action"])
            storage.insert(next_obs, reward, done, info, **act_data)
            logger.collect_env_step_info(info)

        updater.update(policy, storage, logger, envs=envs)

        storage.after_update()

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
            logger.interval_log(update_i, steps_per_update * (update_i + 1))

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
