from typing import Callable, List

import higher
import torch
import torch.nn as nn
from hydra.utils import call, instantiate
from imitation_learning.common.plotting import plot_actions
from imitation_learning.common.utils import (
    extract_transition_batch,
    log_finished_rewards,
)
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchrl.data import TensorDict
from torchrl.envs.utils import step_tensordict


class MetaIRL(nn.Module):
    def __init__(
        self,
        reward: DictConfig,
        inner_updater: DictConfig,
        get_dataset_fn,
        batch_size: int,
        inner_opt: DictConfig,
        reward_opt: DictConfig,
        irl_loss: DictConfig,
        plot_interval: int,
        norm_expert_actions: bool,
        n_inner_iters: int,
        num_steps: int,
        reward_update_freq: int,
        device,
        total_num_updates: int,
        num_envs: int,
        info_keys: List[str],
        use_lr_decay: bool,
        policy_init_fn: Callable[[nn.Module, nn.Module], nn.Module],
        force_num_env_steps_lr_decay: float = -1.0,
        **kwargs,
    ):
        super().__init__()
        self.inner_updater = instantiate(inner_updater)
        self.reward = instantiate(reward).to(device)

        self.dataset = call(get_dataset_fn)
        self.data_loader = DataLoader(self.dataset, batch_size, shuffle=True)

        self.inner_opt = inner_opt
        self.reward_opt = instantiate(reward_opt, params=self.reward.parameters())
        self._n_updates = 0
        self.use_lr_decay = use_lr_decay
        self.policy_init_fn = policy_init_fn

        if force_num_env_steps_lr_decay > 0:
            use_total_num_updates = force_num_env_steps_lr_decay // (
                num_envs * num_steps
            )
        else:
            use_total_num_updates = total_num_updates

        self.lr_scheduler = LambdaLR(
            optimizer=self.reward_opt,
            lr_lambda=lambda x: 1 - (self._n_updates / use_total_num_updates),
        )

        self.irl_loss = instantiate(irl_loss)
        self.data_loader_iter = iter(self.data_loader)

        self.plot_interval = plot_interval
        self.norm_expert_actions = norm_expert_actions
        self.n_inner_iters = n_inner_iters
        self.num_steps = num_steps
        self.reward_update_freq = reward_update_freq
        self.device = device
        self.num_envs = num_envs
        self.info_keys = info_keys
        self._ep_rewards = torch.zeros(num_envs, device=self.device)

    def state_dict(self):
        return {**super().state_dict(), "reward_opt": self.reward_opt.state_dict()}

    def load_state_dict(self, state_dict, should_load_opt):
        opt_state = state_dict.pop("reward_opt")
        if should_load_opt:
            self.reward_opt.load_state_dict(opt_state)
        return super().load_state_dict(state_dict)

    def viz_reward(self, cur_obs=None, action=None, next_obs=None) -> torch.Tensor:
        return self.reward(cur_obs, action, next_obs)

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

        td = TensorDict(source={"observation": expert_obs}, batch_size=[])
        dist = policy.get_action_dist(td)
        pred_actions = dist.mean

        irl_loss_val = self.irl_loss(expert_actions, pred_actions)
        irl_loss_val.backward(retain_graph=True)

        logger.collect_info("irl_loss", irl_loss_val.item())

        if self._n_updates % self.plot_interval == 0:
            plot_actions(
                pred_actions.detach().cpu(),
                expert_actions.detach().cpu(),
                self._n_updates,
                logger.vid_dir,
            )

    @property
    def inner_lr(self):
        return self.inner_opt["lr"]

    def update(self, policy, rollouts, logger, envs):
        self.reward_opt.zero_grad()

        policy = call(self.policy_init_fn, old_policy=policy).to(self.device)
        policy_opt = instantiate(
            self.inner_opt, lr=self.inner_lr, params=policy.parameters()
        )

        # Setup Meta loop
        with higher.innerloop_ctx(
            policy,
            policy_opt,
        ) as (dpolicy, diffopt):
            for inner_i in range(self.n_inner_iters):
                obs, actions, next_obs = extract_transition_batch(rollouts)
                rewards = self.reward(obs, actions, next_obs)

                if inner_i == 0:
                    self._ep_rewards = log_finished_rewards(
                        rollouts, self._ep_rewards, logger
                    )

                # Inner loop policy update
                self.inner_updater.update(dpolicy, rollouts, logger, diffopt)

                if inner_i != self.n_inner_iters - 1:
                    td = rollouts[:, -1]
                    new_rollouts = TensorDict(
                        {},
                        batch_size=[self.num_envs, self.num_steps],
                        device=self.device,
                    )
                    for step_idx in range(self.num_steps):
                        with torch.no_grad():
                            policy.act(td)
                        envs.step(td)

                        new_rollouts[:, step_idx] = td
                        any_env_done = td["done"].any()

                        if any_env_done:
                            td.set("reset_workers", td["done"])
                            envs.reset(tensordict=td)
                            td["next_observation"] = td["observation"]
                            logger.collect_env_step_info(td, self.info_keys)

                        td = step_tensordict(td)
                    rollouts = new_rollouts

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

        if self.use_lr_decay and self.reward_update_freq != -1:
            # Step even if we did not update so we properly decay to 0.
            self.lr_scheduler.step()
            logger.collect_info("reward_lr", self.lr_scheduler.get_last_lr()[0])

        self._n_updates += 1
