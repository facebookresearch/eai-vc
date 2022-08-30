import numpy as np
import torch
import torch.nn as nn
from hydra.utils import call, instantiate
from imitation_learning.common.utils import (
    create_next_obs,
    extract_transition_batch,
    log_finished_rewards,
)
from omegaconf import DictConfig
from rl_utils.common import DictDataset, make_mlp_layers
from torch.utils.data import DataLoader


class NeuralReward(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        reward_hidden_dim,
        cost_take_dim,
        n_hidden_layers,
    ):
        super().__init__()
        self.cost_take_dim = cost_take_dim

        obs_size = obs_shape[0] if cost_take_dim == -1 else cost_take_dim
        self.net = nn.Sequential(
            *make_mlp_layers(obs_size, 1, reward_hidden_dim, n_hidden_layers)
        )

    def forward(self, obs):
        return self.net(obs)


class MaxEntIRL(nn.Module):
    def __init__(
        self,
        reward: DictConfig,
        reward_opt: DictConfig,
        get_dataset_fn,
        batch_size: int,
        num_cost_epochs: int,
        device,
        policy_updater: DictConfig,
        should_update_reward: bool,
        policy,
        num_envs,
        grid_lower=-1.5,
        grid_upper=1.5,
        grid_density=310,
        **kwargs
    ):
        super().__init__()
        self.reward = instantiate(reward).to(device)
        self.policy_updater = instantiate(policy_updater, policy=policy)

        self.demo_obs = call(get_dataset_fn)["next_observations"]

        self.reward_opt = instantiate(reward_opt, params=self.reward.parameters())
        self.num_cost_epochs = num_cost_epochs
        self._ep_rewards = torch.zeros(num_envs, device=device)
        self.should_update_reward = should_update_reward

        # instead of policy samples we use a grid - which we can do in this low dimensional toy example,
        # this will give us "exact" max ent

        self.grid_samples = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(grid_lower, grid_upper, grid_density),
                    torch.linspace(grid_lower, grid_upper, grid_density),
                    indexing="ij",
                ),
                -1,
            )
            .to(device)
            .view(-1, 2)
        )

    def state_dict(self, **kwargs):
        return {
            **super().state_dict(**kwargs),
            "reward_opt": self.reward_opt.state_dict(),
        }

    def load_state_dict(self, state_dict, should_load_opt):
        opt_state = state_dict.pop("reward_opt")
        if should_load_opt:
            self.reward_opt.load_state_dict(opt_state)
        return super().load_state_dict(state_dict)

    def viz_reward(self, cur_obs=None, action=None, next_obs=None) -> torch.Tensor:
        return self.reward(next_obs)

    def update(self, policy, rollouts, logger, **kwargs):
        if self.should_update_reward:
            for _ in range(self.num_cost_epochs):
                reward_samples = self.reward(self.grid_samples)

                # we sample a demo
                reward_demos = self.reward(self.demo_obs)

                # optimize reward
                loss_ME = -(
                    torch.mean(reward_demos)
                    - (
                        torch.logsumexp(reward_samples, dim=0)
                        - np.log(len(reward_samples))
                    )
                )
                self.reward_opt.zero_grad()
                loss_ME.backward()
                self.reward_opt.step()
                logger.collect_info("irl_loss", loss_ME.item())
        else:
            with torch.no_grad():
                _, _, next_obs, _ = extract_transition_batch(rollouts)
                rollouts.rewards = self.reward(next_obs)
                self._ep_rewards = log_finished_rewards(
                    rollouts, self._ep_rewards, logger
                )
            self.policy_updater.update(policy, rollouts, logger)
