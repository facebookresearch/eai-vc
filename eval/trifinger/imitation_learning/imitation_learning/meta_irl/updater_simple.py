from typing import Callable

import higher
import torch
import torch.nn as nn
from hydra.utils import call, instantiate
from imitation_learning.common.plotting import plot_actions
from omegaconf import DictConfig
from rl_helper.common import DictDataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import imitation_learning
import os


class MetaIRL(nn.Module):
    def __init__(
        self,
        reward: DictConfig,
        inner_updater: DictConfig,
        dataset_path: str,
        batch_size: int,
        inner_opt: DictConfig,
        reward_opt: DictConfig,
        irl_loss: DictConfig,
        plot_interval: int,
        norm_expert_actions: bool,
        n_inner_iters: int,
        num_steps: int,
        reward_update_freq: int,
        storage_cfg: DictConfig,
        device,
        total_num_updates: int,
        num_envs: int,
        use_lr_decay: bool,
        lr_decay_speed: int,
        policy_init_fn: Callable[[nn.Module, nn.Module], nn.Module],
        **kwargs,
    ):
        super().__init__()
        self.inner_updater = instantiate(inner_updater)
        self.reward = instantiate(reward).to(device)
        full_dataset_path = os.path.join(imitation_learning.__path__[0], dataset_path)
        self.dataset = DictDataset(
            torch.load(full_dataset_path),
            ["observations", "actions", "rewards", "terminals"],
        )
        self.data_loader = DataLoader(self.dataset, batch_size, shuffle=True)
        self.inner_opt = inner_opt
        self.reward_opt = instantiate(reward_opt, params=self.reward.parameters())
        self._n_updates = 0
        self.use_lr_decay = use_lr_decay
        self.policy_init_fn = policy_init_fn
        self.lr_scheduler = LambdaLR(
            optimizer=self.reward_opt,
            lr_lambda=lambda x: 1
            - (
                self._n_updates
                / (lr_decay_speed * total_num_updates / (num_steps * num_envs))
            ),
        )

        self.irl_loss = instantiate(irl_loss)
        self.data_loader_iter = iter(self.data_loader)

        self.plot_interval = plot_interval
        self.norm_expert_actions = norm_expert_actions
        self.n_inner_iters = n_inner_iters
        self.num_steps = num_steps
        self.reward_update_freq = reward_update_freq
        self.storage_cfg = storage_cfg
        self.device = device
        self.all_rollouts = [
            instantiate(self.storage_cfg, device=self.device)
            for _ in range(self.n_inner_iters - 1)
        ]
        self._ep_rewards = torch.zeros(num_envs, device=self.device)

    def get_reward(self, rollouts):
        obs = next(iter(rollouts.obs.values()))

        cur_obs = obs[:-1]
        masks = rollouts.masks[1:]
        next_obs = (masks * obs[1:]) + ((1 - masks) * rollouts.final_obs)
        actions = rollouts.actions

        return self.reward(cur_obs, actions, next_obs)

    def _irl_loss_step(self, policy, logger):
        expert_batch = next(self.data_loader_iter, None)
        if expert_batch is None:
            self.data_loader_iter = iter(self.data_loader)
            expert_batch = next(self.data_loader_iter, None)
        expert_actions = expert_batch["actions"].to(self.device)
        expert_obs = expert_batch["observations"].to(self.device)
        if self.norm_expert_actions:
            # Clip expert actions to be within [-1,1]. Actions have no effect
            # out of that range
            expert_actions = torch.clamp(expert_actions, -1.0, 1.0)

        dist = policy.get_action_dist(expert_obs, None, None)
        pred_actions = dist.mean

        irl_loss_val = self.irl_loss(expert_actions, pred_actions)
        irl_loss_val.backward(retain_graph=True)

        logger.collect_info("irl_loss", irl_loss_val.item())

        # if self._n_updates % self.plot_interval == 0:
        #     plot_actions(
        #         pred_actions.detach().cpu(),
        #         expert_actions.detach().cpu(),
        #         self._n_updates,
        #         logger.vid_dir,
        #     )

    @property
    def inner_lr(self):
        return self.inner_opt["lr"]

    def _log_ep_rewards(self, rollouts, logger):
        num_steps, num_envs = rollouts.rewards.shape[:2]
        for env_i in range(num_envs):
            for step_i in range(num_steps):
                self._ep_rewards[env_i] += rollouts.rewards[step_i, env_i].item()
                if rollouts.masks[step_i + 1, env_i].item() == 0.0:
                    logger.collect_info(
                        "inferred_episode_reward", self._ep_rewards[env_i].item()
                    )
                    self._ep_rewards[env_i] = 0

    def update(self, policy, rollouts, logger):
        self.reward_opt.zero_grad()

        policy_opt = instantiate(
            self.inner_opt, lr=self.inner_lr, params=policy.parameters()
        )

        # Setup Meta loop
        with higher.innerloop_ctx(
            policy,
            policy_opt,
        ) as (dpolicy, diffopt):
            # Fill in the rewards with the predicted rewards.
            rollouts.rewards = self.get_reward(rollouts)

            self._log_ep_rewards(rollouts, logger)

            # Inner loop policy update
            self.inner_updater.update(
                dpolicy, rollouts, logger, diffopt, rollouts.rewards
            )

            # Compute IRL loss
            self._irl_loss_step(dpolicy, logger)

        if (
            self.reward_update_freq != -1
            and self._n_updates % self.reward_update_freq == 0
        ):
            self.reward_opt.step()
            if hasattr(self.reward, "log"):
                self.reward.log(logger)

        policy.load_state_dict(dpolicy.state_dict())

        if self.use_lr_decay:
            # Step even if we did not update so we properly decay to 0.
            self.lr_scheduler.step()
            logger.collect_info("reward_lr", self.lr_scheduler.get_last_lr()[0])

        self._n_updates += 1
