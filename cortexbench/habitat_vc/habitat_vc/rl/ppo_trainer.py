#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import tqdm
from typing import Any, Dict, List

import torch
from torch import nn
import wandb

from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.ddppo.ddp_utils import rank0_only
from habitat_baselines.utils.common import (
    action_to_velocity_control,
    batch_obs,
    get_checkpoint_id,
)

from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer

from habitat_vc.rl.ppo import MDDPPO, MPPO
import habitat_vc.utils as utils


@baseline_registry.register_trainer(name="mddppo")
@baseline_registry.register_trainer(name="mppo")
class ModifiedPPOTrainer(PPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.wandb_initialized = False

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config, observation_space, self.policy_action_space
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = (MDDPPO if self._is_distributed else MPPO)(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            encoder_lr=ppo_cfg.encoder_lr,
            wd=ppo_cfg.wd,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    @rank0_only
    def _training_log(self, writer, losses: Dict[str, float], prev_time: int = 0):
        if self.wandb_initialized == False:
            utils.setup_wandb(self.config, train=True)
            self.wandb_initialized = True

        deltas = {
            k: ((v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item())
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        wandb.log(
            {"train/reward": deltas["reward"] / deltas["count"]},
            step=self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }
        # To solve a wandb related error
        metrics = {f"train/{k}": v for k, v in metrics.items() if v >= 0 and v < 100}
        if len(metrics) > 0:
            wandb.log(metrics, step=self.num_steps_done)

        wandb_losses = {f"train/{k}": v for k, v in losses.items()}
        wandb.log(wandb_losses, step=self.num_steps_done)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    self.num_steps_done / ((time.time() - self.t_start) + prev_time),
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        utils.setup_wandb(self.config, train=False)

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
            # evaluate single checkpoint
            proposed_index = get_checkpoint_id(self.config.EVAL_CKPT_PATH_DIR)

            if proposed_index is not None:
                ckpt_idx = proposed_index
            else:
                ckpt_idx = 0

            self._eval_checkpoint(
                self.config.EVAL_CKPT_PATH_DIR,
                checkpoint_index=ckpt_idx,
            )

        else:
            # evaluate multiple checkpoints in order
            eval_iter_filename = os.path.join(
                self.config.TENSORBOARD_DIR,
                "eval_iter_" + str(self.config.EVAL.SPLIT) + ".txt",
            )

            if os.path.exists(eval_iter_filename):
                with open(eval_iter_filename, "r") as file:
                    prev_ckpt_ind = file.read().rstrip("\n")
                    prev_ckpt_ind = int(prev_ckpt_ind)
            else:
                prev_ckpt_ind = -1

            while True:
                current_ckpt = None
                while current_ckpt is None:
                    current_ckpt, current_ckpt_idx = utils.poll_checkpoint_folder(
                        self.config.EVAL_CKPT_PATH_DIR,
                        prev_ckpt_ind,
                        self.config.EVAL.EVAL_FREQ,
                        self.config.NUM_CHECKPOINTS,
                    )
                    time.sleep(2)  # sleep for 2 secs before polling again

                logger.info(f"=======current_ckpt: {current_ckpt}=======")
                prev_ckpt_ind = current_ckpt_idx
                with open(eval_iter_filename, "w") as file:
                    file.write(str(prev_ckpt_ind))

                self._eval_checkpoint(
                    checkpoint_path=current_ckpt,
                    checkpoint_index=prev_ckpt_ind,
                )

                if self.config.NUM_CHECKPOINTS - 1 == prev_ckpt_ind:
                    break

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

        if self.using_velocity_ctrl:
            self.policy_action_space = self.envs.action_spaces[0]["VELOCITY_CONTROL"]
            action_shape = (2,)
            action_type = torch.float
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = (1,)
            action_type = torch.long

        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.net.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=action_type,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while len(stats_episodes) < number_of_eval_episodes and self.envs.num_envs > 0:
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if self.using_velocity_ctrl:
                step_data = [
                    action_to_velocity_control(a) for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(self._extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        utils.generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=5,
                        )
                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = utils.observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    frame = utils.add_info_to_image(frame, infos[i])
                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        metrics = {f"eval/{k}": v for k, v in aggregated_stats.items()}
        if len(metrics) > 0:
            wandb.log(metrics, step=step_id)

        self.envs.close()
