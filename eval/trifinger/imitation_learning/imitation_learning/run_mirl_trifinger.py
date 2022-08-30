import os.path as osp
from typing import Dict
import numpy as np
from collections import defaultdict
import datetime

import hydra
import torch
from hydra.utils import instantiate as hydra_instantiate

from rl_utils.common import Evaluator, set_seed
from rl_utils.envs import create_vectorized_envs
from rl_utils.logging import Logger, WbLogger


import imitation_learning.common.trifinger_envs as trifinger_envs

from imitation_learning.policy_opt.policy import Policy
from imitation_learning.policy_opt.storage import RolloutStorage


DEMO_GOAL = np.array([-0.012, -0.0, 0.089, 0.014, 0.005, 0.128, 0.015, -0.014, 0.152])


class TrifingerEvaluator(Evaluator):
    def evaluate(self, policy, num_episodes, eval_i) -> Dict[str, float]:
        self._clear_save_trajs()

        if isinstance(policy, torch.nn.Module):
            device = next(policy.parameters()).device
        else:
            device = torch.device("cpu")

        num_envs = self._envs.num_envs
        obs = self._envs.reset()
        rnn_hxs = torch.zeros(num_envs, self._rnn_hxs_dim).to(device)
        eval_masks = torch.zeros(num_envs, 1, device=device)

        accum_stats = defaultdict(list)

        for n in range(5):

            act_data = policy.act(obs, rnn_hxs, eval_masks, deterministic=True)
            next_obs, rewards, done, infos = self._envs.step(
                act_data["action"].detach()
            )
            rnn_hxs = act_data["recurrent_hidden_states"]

            obs = next_obs

        dist_to_goal = [
            np.mean((x["desired_goal"] - x["achieved_goal"]) ** 2) for x in infos
        ]
        dist_to_demogoal = [
            np.mean((DEMO_GOAL - x["achieved_goal"]) ** 2) for x in infos
        ]
        success = np.array(np.array(dist_to_goal) < 0.001, dtype=float)
        successdemo = np.array(np.array(dist_to_demogoal) < 0.001, dtype=float)
        accum_stats["eval_dist2goal"] = dist_to_goal
        accum_stats["eval_dist2demogoal"] = dist_to_demogoal
        accum_stats["eval_success"] = success
        accum_stats["eval_successdemo"] = successdemo

        return {k: np.mean(v) for k, v in accum_stats.items()}


@hydra.main(config_path="config/airl", config_name="trifinger")
def main(cfg) -> Dict[str, float]:
    set_seed(cfg.seed)
    d = datetime.datetime.today()
    # date_id = "%i%i" % (d.month, d.day)
    # group_name = cfg.logger.group_name
    # cfg.logger.group_name = f"{date_id}-{group_name}"

    def create_trifinger_env(seed):
        np.random.seed(seed)
        env = trifinger_envs.CausalWorldReacherWrapper(
            start_state_noise=cfg.env_settings.start_state_noise,
            skip_frame=10,
            max_ep_horizon=5,
        )
        return env

    envs = create_vectorized_envs(
        cfg.env_name, cfg.num_envs, seed=cfg.seed, create_env_fn=create_trifinger_env
    )
    #
    device = torch.device("cpu")
    cfg.obs_shape = envs.observation_space.shape
    cfg.action_dim = envs.action_space.shape[0]
    cfg.action_is_discrete = False
    logger: Logger = hydra_instantiate(cfg.logger, full_cfg=cfg)
    # cfg.logger.wb_entity = "fmeier"
    # cfg.logger.wb_proj_name = "irl-trifinger"

    # logger: Logger = WbLogger(wb_proj_name=cfg.logger.wb_proj_name, wb_entity=cfg.logger.wb_entity,
    #                           run_name=cfg.logger.run_name, group_name=cfg.logger.group_name, seed=cfg.logger.seed,
    #                           log_dir=cfg.logger.log_dir, vid_dir=cfg.logger.vid_dir, save_dir=cfg.logger.save_dir,
    #                           smooth_len=cfg.logger.smooth_len, full_cfg=cfg)

    # hydra_instantiate(cfg.logger, full_cfg=cfg)

    storage: RolloutStorage = hydra_instantiate(cfg.storage, device=device)
    policy: Policy = hydra_instantiate(cfg.policy)
    updater = hydra_instantiate(cfg.policy_updater, policy=policy)

    evaluator: Evaluator = hydra_instantiate(
        cfg.evaluator, envs=envs, vid_dir=logger.vid_path, updater=updater
    )

    obs = envs.reset()
    storage.init_storage(obs)

    steps_per_update = cfg.num_steps * cfg.num_envs
    num_updates = int(cfg.num_env_steps) // steps_per_update

    start_update = 0
    if cfg.load_checkpoint is not None:
        ckpt = torch.load(cfg.load_checkpoint)
        updater.load_state_dict(ckpt["updater"])
        if cfg.load_policy:
            policy.load_state_dict(ckpt["policy"])
        if cfg.resume_training:
            start_update = ckpt["update_i"] + 1

    eval_info = {"run_name": logger.run_name}

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

        updater.update(policy, storage, logger)

        storage.after_update()

        if cfg.eval_interval != -1 and (
            update_i % cfg.eval_interval == 0 or is_last_update
        ):
            eval_result = evaluator.evaluate(policy, cfg.num_eval_episodes, update_i)
            eval_info.update(eval_result)
            logger.collect_infos(eval_result, "eval.", no_rolling_window=True)

        if cfg.log_interval != -1 and (
            update_i % cfg.log_interval == 0 or is_last_update
        ):
            logger.interval_log(update_i, steps_per_update * (update_i + 1))

        if cfg.save_interval != -1 and (
            update_i % cfg.save_interval == 0 or is_last_update
        ):
            save_name = osp.join(
                logger.save_path, f"{cfg.train_or_eval}_ckpt.{update_i}.pth"
            )
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
    envs.close()
    return eval_info


if __name__ == "__main__":
    main()
