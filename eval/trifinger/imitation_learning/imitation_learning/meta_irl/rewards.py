from enum import Enum, auto

import torch
import torch.nn as nn
from hydra.utils import instantiate
from rl_utils.common import make_mlp_layers


class RewardInputType(Enum):
    ACTION = auto()
    NEXT_STATE = auto()
    CUR_NEXT_STATE = auto()


def full_reset_init(old_policy, policy_cfg, **kwargs):
    return instantiate(policy_cfg)


def reg_init(old_policy, **kwargs):
    return old_policy


class StructuredReward(nn.Module):
    def __init__(self, obs_shape, **kwargs):
        super().__init__()
        self.center = nn.Parameter(torch.randn(obs_shape[0]))

    def forward(self, X):
        return -1.0 * ((X - self.center) ** 2).mean(-1, keepdims=True)

    def log(self, logger):
        for i, center_val in enumerate(self.center):
            logger.collect_info(f"reward_{i}", center_val.item())


class GtReward(nn.Module):
    def __init__(
        self,
    ):
        pass

    def forward(self, cur_obs=None, actions=None, next_obs=None):
        cur_dist = torch.linalg.norm(cur_obs, dim=-1)
        reward = torch.full(cur_dist.shape, -self._slack)
        assign = -self._slack * cur_dist
        should_give_reward = cur_dist < self._reward_thresh
        reward[should_give_reward] = assign[should_give_reward]
        return reward


class NeuralReward(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        reward_hidden_dim,
        reward_type,
        cost_take_dim,
        n_hidden_layers,
        include_tanh,
    ):
        super().__init__()
        self.reward_type = RewardInputType[reward_type]
        self.cost_take_dim = cost_take_dim

        obs_size = obs_shape[0] if cost_take_dim == -1 else abs(cost_take_dim)

        if self.reward_type == RewardInputType.ACTION:
            input_size = obs_size + action_dim
        elif self.reward_type == RewardInputType.NEXT_STATE:
            input_size = obs_size
        elif self.reward_type == RewardInputType.CUR_NEXT_STATE:
            input_size = obs_size + obs_size

        net_layers = make_mlp_layers(input_size, 1, reward_hidden_dim, n_hidden_layers)
        if include_tanh:
            net_layers.append(nn.Tanh())
        self.net = nn.Sequential(*net_layers)

    def forward(self, cur_obs=None, actions=None, next_obs=None):
        if self.cost_take_dim != -1:
            if cur_obs is not None:
                cur_obs = cur_obs[:, :, self.cost_take_dim :]
            if next_obs is not None:
                next_obs = next_obs[:, :, self.cost_take_dim :]

        if self.reward_type == RewardInputType.ACTION:
            inputs = torch.cat([cur_obs, actions], dim=-1)
        elif self.reward_type == RewardInputType.NEXT_STATE:
            inputs = torch.cat([next_obs], dim=-1)
        elif self.reward_type == RewardInputType.CUR_NEXT_STATE:
            inputs = torch.cat([cur_obs, next_obs], dim=-1)

        return self.net(inputs)
