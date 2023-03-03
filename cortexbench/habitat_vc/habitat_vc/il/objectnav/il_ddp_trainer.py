#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
import numpy as np
import torch
import wandb

from collections import defaultdict, deque
from typing import DefaultDict, Optional

from torch import distributed as distrib
from torch import nn as nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_resume_state,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.utils.common import batch_obs, linear_decay
from habitat_baselines.utils.env_utils import construct_envs

from habitat_vc.il.objectnav.algos.agent import DDPILAgent
from habitat_vc.il.objectnav.il_trainer import ILEnvTrainer
from habitat_vc.il.objectnav.rollout_storage import RolloutStorage
from habitat_vc.il.objectnav.custom_baseline_registry import custom_baseline_registry
import habitat_vc.utils as utils


@baseline_registry.register_trainer(name="ddp-il-trainer")
class ILEnvDDPTrainer(ILEnvTrainer):
    # DD-PPO cuts rollouts short to mitigate the straggler effect
    # This, in theory, can cause some rollouts to be very short.
    # All rollouts contributed equally to the loss/model-update,
    # thus very short rollouts can be problematic.  This threshold
    # limits the how short a short rollout can be as a fraction of the
    # max rollout length
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config: Optional[Config] = None) -> None:
        interrupted_state = load_resume_state(config)
        if interrupted_state is not None:
            config = interrupted_state["config"]

        super().__init__(config)

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

        # Load pretrained state
        if self.config.IL.BehaviorCloning.pretrained:
            pretrained_state = torch.load(
                self.config.IL.BehaviorCloning.pretrained_weights, map_location="cpu"
            )
            logger.info("Loading pretrained state")

        if self.config.IL.BehaviorCloning.pretrained:
            missing_keys = self.policy.load_state_dict(
                {
                    k.replace("model.", ""): v
                    for k, v in pretrained_state["state_dict"].items()
                },
                strict=False,
            )
            logger.info("Loading checkpoint missing keys: {}".format(missing_keys))

        self.agent = DDPILAgent(
            model=self.policy,
            num_envs=self.envs.num_envs,
            num_mini_batch=il_cfg.num_mini_batch,
            lr=il_cfg.lr,
            encoder_lr=il_cfg.encoder_lr,
            eps=il_cfg.eps,
            wd=il_cfg.wd,
            max_grad_norm=il_cfg.max_grad_norm,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for DD-PPO.

        Returns:
            None
        """
        self.local_rank, tcp_store = init_distrib_slurm(self.config.IL.distrib_backend)
        add_signal_handlers()

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )
        SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
        interrupted_state_file = os.path.join(
            self.config.CHECKPOINT_FOLDER, "{}.pth".format(SLURM_JOBID)
        )

        interrupted_state = load_resume_state(self.config)
        if interrupted_state is not None:
            logger.info("Overriding current config with interrupted state config")
            self.config = interrupted_state["config"]

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore("rollout_tracker", tcp_store)
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += self.world_rank * self.config.NUM_PROCESSES
        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            workers_ignore_signals=True,
        )

        logger.info(
            "[ train_loader has {} samples ]".format(self.envs.count_episodes())
        )

        il_cfg = self.config.IL.BehaviorCloning
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER) and self.world_rank == 0:
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(il_cfg, self.config.MODEL)
        self.agent.init_distributed(find_unused_params=True)
        self.agent.train()

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )

            if self.wandb_initialized == False:
                utils.setup_wandb(self.config, train=True)
                self.wandb_initialized = True

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        obs_space = self.obs_space

        # To handle LSTM input
        num_rnn_layer_multiplier = (
            2 if self.config.MODEL.STATE_ENCODER.rnn_type == "LSTM" else 1
        )
        rollouts = RolloutStorage(
            il_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.envs.action_spaces[0],
            self.config.MODEL.STATE_ENCODER.hidden_size,
            num_recurrent_layers=self.config.MODEL.STATE_ENCODER.num_recurrent_layers
            * num_rnn_layer_multiplier,
        )
        rollouts.to(self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
        )
        window_episode_stats: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=il_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps: int = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),  # type: ignore
        )
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(interrupted_state["optim_state"])
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with (
            TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs)
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")
                self.current_update = update

                if update > 0 and il_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                if update > 0 and il_cfg.use_linear_clip_decay:
                    self.agent.clip_param = il_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    if self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        save_resume_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            ),
                            interrupted_state_file,
                        )

                    requeue_job()
                    return

                count_steps_delta = 0
                self.agent.eval()
                profiling_wrapper.range_push("rollouts loop")
                for step in range(il_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps_delta += delta_steps

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step >= il_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        il_cfg.sync_frac * self.world_size
                    ):
                        break
                profiling_wrapper.range_pop()  # rollouts loop

                num_rollouts_done_store.add("num_done", 1)
                # logger.info("update: {}".format(update))

                self.agent.train()
                (delta_pth_time, total_loss) = self._update_agent(il_cfg, rollouts)
                pth_time += delta_pth_time

                stats_ordering = sorted(running_episode_stats.keys())
                stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                )
                distrib.all_reduce(stats)

                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone())

                stats = torch.tensor(
                    [total_loss, count_steps_delta],
                    device=self.device,
                )
                distrib.all_reduce(stats)
                count_steps += int(stats[1].item())

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                    losses = [
                        stats[0].item() / self.world_size,
                    ]
                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
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
                        f"train/{k}": v
                        for k, v in metrics.items()
                        if v >= 0 and v < 100
                    }
                    if len(metrics) > 0:
                        wandb.log(metrics, step=count_steps)

                    wandb.log(
                        {f"train/{k}": l for l, k in zip(losses, ["action_loss"])},
                        step=count_steps,
                    )

                    # log stats
                    if update > 0 and update % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}\tloss: {:.3f}".format(
                                update,
                                count_steps / ((time.time() - t_start) + prev_time),
                                losses[0],
                            )
                        )

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s"
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
                            f"ckpt.{count_checkpoints}.pth",
                            dict(step=count_steps),
                        )
                        count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()
