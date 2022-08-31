from typing import Any, Dict

import torch
import torch.nn as nn
from hydra.utils import instantiate as hydra_instantiate
from imitation_learning.policy_opt.policy import Policy


class REINFORCE(nn.Module):
    def __init__(
        self,
        gamma: float,
        max_grad_norm: float,
        optimizer_params: Dict[str, Any],
        policy: Policy,
        **kwargs,
    ):
        super().__init__()
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        self.optimizer: torch.optim.Optimizer = hydra_instantiate(
            optimizer_params, params=policy.parameters()
        )

    def state_dict(self):
        ret = super().state_dict()
        return {**ret, "opt": self.optimizer.state_dict()}

    def update(
        self,
        policy,
        storage,
        logger,
        envs,
    ):
        returns = self.compute_derived(
            policy,
            storage.rewards,
            storage.masks,
            storage.bad_masks,
        )

        data_gen = storage.data_generator(1, returns=returns[:-1])
        sample = next(iter(data_gen))
        ac_eval = policy.evaluate_actions(
            sample["observation"],
            sample["hxs"],
            sample["mask"],
            sample["action"],
        )

        loss = (-sample["returns"] * ac_eval["log_prob"]).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        logger.collect_info("action_loss", loss.mean().item())
        logger.collect_info("dist_entropy", ac_eval["dist_entropy"].mean().item())

    def compute_derived(
        self,
        policy,
        rewards,
        masks,
        bad_masks,
    ):
        num_steps, num_envs = rewards.shape[:2]
        returns = torch.zeros(num_steps + 1, num_envs, 1, device=rewards.device)
        for step in reversed(range(rewards.size(0))):
            returns[step] = (
                returns[step + 1] * self.gamma * masks[step + 1] + rewards[step]
            )
        return returns
