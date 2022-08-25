from typing import Tuple

import torch
import torch.nn as nn
from rl_utils.common import make_mlp_layers


class AirlDiscriminator(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int],
        action_dim: int,
        reward_hidden_dim: int,
        cost_take_dim: int,
        n_hidden_layers: int,
        use_shaped_reward: bool,
        gamma: float,
        airl_reward_bonus: float,
    ):
        super().__init__()
        self.cost_take_dim = cost_take_dim
        state_size = obs_shape[0] if cost_take_dim == -1 else cost_take_dim

        self.g = nn.Sequential(
            *make_mlp_layers(state_size, 1, reward_hidden_dim, n_hidden_layers)
        )
        self.h = nn.Sequential(
            *make_mlp_layers(state_size, 1, reward_hidden_dim, n_hidden_layers)
        )
        self.use_shaped_reward = use_shaped_reward
        self.gamma = gamma
        self.airl_reward_bonus = airl_reward_bonus

    def f(self, cur_obs, next_obs, masks, force_no_shaped=False, **kwargs):
        rs = self.g(cur_obs)
        if self.use_shaped_reward and not force_no_shaped:
            vs = self.h(cur_obs)
            next_vs = self.h(next_obs)
            return rs + (self.gamma * masks * next_vs) - vs
        else:
            return rs

    def forward(self, cur_obs, next_obs, actions, masks, policy, **kwargs):
        log_p = self.f(cur_obs, next_obs, masks)

        with torch.no_grad():
            log_q = policy.evaluate_actions(cur_obs, {}, masks, actions)["log_prob"]

        return log_p - log_q

    def get_reward(
        self,
        cur_obs,
        next_obs,
        masks=None,
        actions=None,
        policy=None,
        viz_reward=False,
        **kwargs
    ):
        log_p = self.f(cur_obs, next_obs, masks=masks, force_no_shaped=viz_reward)
        if viz_reward:
            return log_p

        with torch.no_grad():
            log_q = policy.evaluate_actions(cur_obs, {}, masks, actions)["log_prob"]

        logits = log_p - (self.airl_reward_bonus * log_q)
        s = torch.sigmoid(logits)
        eps = 1e-20
        return (s + eps).log() - (1 - s + eps).log()
