from typing import Any, Dict

import torch
import torch.nn as nn
from hydra.utils import instantiate as hydra_instantiate
from imitation_learning.policy_opt.policy import Policy
from torchrl.data import TensorDict
from torchrl.trainers import BatchSubSampler
from torchrl.objectives import SACLoss
from rl_utils.common import set_seed


class SAC:
    def __init__(
        self,
        gamma: float,
        alpha_init: float,
        min_alpha: float,
        max_alpha: float,
        fixed_alpha: bool,
        max_grad_norm: float,
        num_mini_batch: int,
        num_epochs: int,
        optimizer_params: Dict[str, Any],
        num_steps: int,
        num_envs: int,
        policy: Policy,
        **kwargs,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha_init = alpha_init
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.fixed_alpha = fixed_alpha
        self.max_grad_norm = max_grad_norm
        self.num_mini_batch = num_mini_batch
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_envs = num_envs

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
        loss_module = SACLoss(
            actor_network=policy.actor,
            qvalue_network=policy.qvalue_net,
            value_network=policy.value_net,
            alpha_init=self.alpha_init,
            min_alpha=self.min_alpha,
            max_alpha=self.max_alpha,
            fixed_alpha=self.fixed_alpha,
            target_entropy=policy.target_entropy,
        )

        data_gen = BatchSubSampler(
            batch_size=self.num_envs * self.num_steps // self.num_mini_batch
        )
        batch_size = storage.batch_size
        storage_view = storage.view(batch_size[0] * batch_size[1])

        for _ in range(self.num_epochs):
            data = data_gen(storage_view)
            data = data.to_tensordict()

            loss = loss_module(data)
            loss_actor, loss_qvalue, loss_value, loss_alpha = (
                loss["loss_actor"],
                loss["loss_qvalue"],
                loss["loss_value"],
                loss["loss_alpha"],
            )

            total_loss = loss_actor + loss_qvalue + loss_value + loss_alpha

            self.opt.zero_grad()
            total_loss.backward()

            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            self.opt.step()

            logger.collect_info("actor_loss", loss_actor.mean().item())
            logger.collect_info("qvalue_loss", loss_qvalue.mean().item())
            logger.collect_info("value_loss", loss_value.mean().item())
            logger.collect_info("alpha_loss", loss_alpha.mean().item())
