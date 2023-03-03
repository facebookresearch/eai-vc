#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import wandb

from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import tqdm

from numpy import ndarray
from torch.optim.lr_scheduler import LambdaLR
from torch import Tensor

from habitat import Config, logger
from habitat.core.env import Env, RLEnv
from habitat.core.vector_env import VectorEnv
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image

from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    linear_decay,
    get_checkpoint_id,
)
from habitat_baselines.utils.env_utils import construct_envs

from habitat_vc.il.objectnav.algos.agent import ILAgent
from habitat_vc.il.objectnav.rollout_storage import RolloutStorage
from habitat_vc.il.objectnav.custom_baseline_registry import custom_baseline_registry

import habitat_vc.utils as utils


@baseline_registry.register_trainer(name="il-trainer")
class ILEnvTrainer(BaseRLTrainer):
    r"""Trainer class for behavior cloning."""
    supported_tasks = ["ObjectNav-v1"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []
        self.wandb_initialized = False
        if config is not None:
            logger.info(f"config: {config}")

    def _setup_actor_critic_agent(self, il_cfg: Config, model_config: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        observation_space = self.envs.observation_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.obs_space = observation_space

        model_config.defrost()
        model_config.TORCH_GPU_ID = self.config.TORCH_GPU_ID
        model_config.freeze()

        policy = custom_baseline_registry.get_policy(self.config.IL.POLICY.name)
        self.policy = policy.from_config(
            self.config, observation_space, self.envs.action_spaces[0]
        )
        self.policy.to(self.device)

        self.agent = ILAgent(
            model=self.policy,
            num_envs=self.envs.num_envs,
            num_mini_batch=il_cfg.num_mini_batch,
            lr=il_cfg.lr,
            encoder_lr=il_cfg.encoder_lr,
            eps=il_cfg.eps,
            wd=il_cfg.wd,
            max_grad_norm=il_cfg.max_grad_norm,
        )

    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {
        "top_down_map",
        "collisions.is_collision",
        "room_visitation_map",
    }

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict[str, Any]) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(v).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    @staticmethod
    def _pause_envs(
        envs_to_pause: List[int],
        envs: Union[VectorEnv, RLEnv, Env],
        test_recurrent_hidden_states: Tensor,
        not_done_masks: Tensor,
        current_episode_reward: Tensor,
        prev_actions: Tensor,
        batch: Dict[str, Tensor],
        rgb_frames: Union[List[List[Any]], List[List[ndarray]]],
        episode_length: ndarray,
    ) -> Tuple[
        Union[VectorEnv, RLEnv, Env],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Dict[str, Tensor],
        List[List[Any]],
        ndarray,
    ]:
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[:, state_index]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]
            episode_length = episode_length[state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
            episode_length,
        )

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()

        # fetch actions and environment state from replay buffer
        next_actions = rollouts.get_next_actions()
        actions = next_actions.long().unsqueeze(-1)
        step_data = [a.item() for a in next_actions.long().to(device="cpu")]

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        profiling_wrapper.range_pop()  # compute actions

        outputs = self.envs.step(step_data)
        observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward  # type: ignore
        running_episode_stats["count"] += 1 - masks  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v  # type: ignore

        current_episode_reward *= masks

        rollouts.insert(
            batch,
            actions,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()

        total_loss, rnn_hidden_states = self.agent.update(rollouts)

        rollouts.after_update(rnn_hidden_states)

        return (
            time.time() - t_update_model,
            total_loss,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))

        il_cfg = self.config.IL.BehaviorCloning
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(il_cfg, self.config.MODEL)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        if self.wandb_initialized == False:
            utils.setup_wandb(self.config, train=True)
            self.wandb_initialized = True

        # To handle LSTM input
        num_rnn_layer_multiplier = (
            2 if self.config.MODEL.STATE_ENCODER.rnn_type == "LSTM" else 1
        )
        rollouts = RolloutStorage(
            il_cfg.num_steps,
            self.envs.num_envs,
            self.obs_space,
            self.envs.action_spaces[0],
            self.config.MODEL.STATE_ENCODER.hidden_size,
            self.config.MODEL.STATE_ENCODER.num_recurrent_layers
            * num_rnn_layer_multiplier,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=il_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps: int = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),  # type: ignore
        )
        self.possible_actions = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                self.current_update = update

                if il_cfg.use_linear_lr_decay and update > 0:
                    lr_scheduler.step()  # type: ignore

                if il_cfg.use_linear_clip_decay and update > 0:
                    self.agent.clip_param = il_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                profiling_wrapper.range_push("rollouts loop")
                for _step in range(il_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps
                profiling_wrapper.range_pop()  # rollouts loop

                (delta_pth_time, total_loss) = self._update_agent(il_cfg, rollouts)
                pth_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                wandb.log(
                    {"train/reward": deltas["reward"] / deltas["count"]},
                    step=count_steps,
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                # To solve a wandb related error
                metrics = {
                    f"train/{k}": v for k, v in metrics.items() if v >= 0 and v < 100
                }
                if len(metrics) > 0:
                    wandb.log(metrics, step=count_steps)

                losses = [total_loss]
                losses = {f"train/{k}": l for l, k in zip(losses, ["action_loss"])}
                wandb.log(losses, step=count_steps)

                # log stats
                if update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\tloss: {:.3f}".format(
                            update, count_steps / (time.time() - t_start), total_loss
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(update, env_time, pth_time, count_steps)
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

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

        ckpt_path = os.path.join(
            self.config.CHECKPOINT_FOLDER, self.config.EVAL_CKPT_PATH_DIR
        )
        if os.path.isfile(ckpt_path):
            # evaluate single checkpoint
            proposed_index = get_checkpoint_id(ckpt_path)

            if proposed_index is not None:
                ckpt_idx = proposed_index
            else:
                ckpt_idx = 0

            self._eval_checkpoint(
                ckpt_path,
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
                prev_ckpt_ind = self.config.EVAL.FIRST_CHECKPOINT - 1

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
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            conf = ckpt_dict["config"]
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        il_cfg = config.IL.BehaviorCloning

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 500
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(il_cfg, config.MODEL)

        self.agent.load_state_dict(ckpt_dict["state_dict"], strict=True)
        self.policy = self.agent.model
        self.policy.eval()

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)

        # To handle LSTM input
        num_rnn_layer_multiplier = (
            2 if self.config.MODEL.STATE_ENCODER.rnn_type == "LSTM" else 1
        )
        test_recurrent_hidden_states = torch.zeros(
            config.MODEL.STATE_ENCODER.num_recurrent_layers * num_rnn_layer_multiplier,
            config.NUM_PROCESSES,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(config.NUM_PROCESSES, 1, device=self.device)
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        current_episode_steps = torch.zeros(self.envs.num_envs, 1, device=self.device)

        rgb_frames = [
            [] for _ in range(config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        episode_length = np.zeros(config.NUM_PROCESSES, dtype=np.int32)
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = config.TEST_EPISODE_COUNT
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
        episode_meta = []
        while len(stats_episodes) < number_of_eval_episodes and self.envs.num_envs > 0:
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    logits,
                    test_recurrent_hidden_states,
                    dist_entropy,
                ) = self.policy(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                )

                actions = torch.argmax(logits, dim=1)
                prev_actions.copy_(actions.unsqueeze(1))  # type: ignore

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device=self.device
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
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(self._extract_scalars_from_info(infos[i]))
                    # episode_stats["episode_length"] = episode_length[i]
                    current_episode_reward[i] = 0
                    logger.info(
                        "Success: {}, SPL: {}, episode length: {}".format(
                            episode_stats["success"],
                            episode_stats["spl"],
                            episode_length[i],
                        )
                    )
                    episode_meta.append(
                        {
                            "scene_id": current_episodes[i].scene_id,
                            "episode_id": current_episodes[i].episode_id,
                            "metrics": episode_stats,
                        }
                    )
                    utils.write_json(episode_meta, self.config.EVAL.meta_file)

                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                        )

                        rgb_frames[i] = []
                        episode_length[i] = 0

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image({"rgb": batch["rgb"][i]}, infos[i])
                    rgb_frames[i].append(frame)
                episode_length[i] += 1
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                episode_length,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                episode_length,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
        logger.info("Checkpoint path: {}".format(checkpoint_path))

        step_id = int(checkpoint_index)
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = int(ckpt_dict["extra_state"]["step"])

        wandb.log({"eval/average reward": aggregated_stats["reward"]}, step=step_id)
        metrics = {f"eval/{k}": v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            wandb.log(metrics, step=step_id)

        utils.write_json(episode_meta, self.config.EVAL.meta_file)

        self.envs.close()
