from typing import Any, Dict

import torch
import torch.nn as nn
from hydra.utils import instantiate as hydra_instantiate
from imitation_learning.policy_opt.policy import Policy
from torchrl.trainers import BatchSubSampler
from torchrl.objectives import PPOLoss, ClipPPOLoss, GAE, TDEstimate


class PPO:
    def __init__(
        self,
        use_gae: bool,
        gae_lambda: float,
        gamma: float,
        use_clipped_value_loss: bool,
        clip_param: bool,
        value_loss_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        num_mini_batch: int,
        num_epochs: int,
        optimizer_params: Dict[str, Any],
        num_envs: int,
        num_steps: int,
        policy: Policy,
        **kwargs,
    ):
        super().__init__()
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_mini_batch = num_mini_batch
        self.num_epochs = num_epochs
        self.num_envs = num_envs
        self.num_steps = num_steps

        self.opt: torch.optim.Optimizer = hydra_instantiate(
            optimizer_params, params=policy.parameters()
        )

    def state_dict(self):
        return {"opt": self.opt.state_dict()}

    def load_state_dict(self, state_dict, should_load_opt):
        opt_state = state_dict.pop("opt")
        if should_load_opt:
            self.opt.load_state_dict(opt_state)

    def update(
        self,
        policy,
        storage,
        logger,
        **kwargs,
    ):
        if self.use_gae:
            advantage_module = GAE(
                gamma=self.gamma,
                lmbda=self.gae_lambda,
                value_network=policy.critic,
                average_rewards=True,
                gradient_mode=True,
            )
        else:
            advantage_module = TDEstimate(
                gamma=self.gamma,
                value_network=policy.critic,
                average_rewards=True,
                gradient_mode=True,
            )

        # storage = advantage_module(storage)
        if self.use_clipped_value_loss:
            loss_module = ClipPPOLoss(
                policy.actor,
                policy.critic,
                entropy_coef=self.entropy_coef,
                critic_coef=self.value_loss_coef,
                gamma=self.gamma,
                clip_epsilon=self.clip_param,
            )
        else:
            loss_module = PPOLoss(
                policy.actor,
                policy.critic,
                entropy_coef=self.entropy_coef,
                critic_coef=self.value_loss_coef,
                gamma=self.gamma,
            )

        data_gen = BatchSubSampler(
            batch_size=self.num_envs * self.num_steps // self.num_mini_batch
        )

        for _ in range(self.num_epochs):
            data = data_gen(storage).to_tensordict()
            data = advantage_module(data)
            loss = loss_module(data)
            loss_objective, loss_critic, loss_entropy = (
                loss["loss_objective"],
                loss["loss_critic"],
                loss.get("loss_entropy", 0),
            )
            total_loss = loss_objective + loss_critic + loss_entropy

            self.opt.zero_grad()
            total_loss.backward()

            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            self.opt.step()

            loss_e = (
                loss_entropy
                if type(loss_entropy) is int
                else loss_entropy.mean().item()
            )
            logger.collect_info("value_loss", loss_critic.mean().item())
            logger.collect_info("action_loss", loss_objective.mean().item())
            logger.collect_info("dist_entropy", loss_e)
