#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.rl.ppo.ppo import PPO
from torch import optim as optim


class MPPO(PPO):
    """PPO with weight decay."""

    def __init__(
        self,
        actor_critic: Policy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        encoder_lr: Optional[float] = None,
        wd: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
        use_normalized_advantage: bool = True,
    ) -> None:
        super().__init__(
            actor_critic=actor_critic,
            clip_param=clip_param,
            ppo_epoch=ppo_epoch,
            num_mini_batch=num_mini_batch,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            use_normalized_advantage=use_normalized_advantage,
        )

        # use different lr for visual encoder and other networks
        visual_encoder_params, other_params = [], []
        for name, param in actor_critic.named_parameters():
            if param.requires_grad:
                if (
                    "net.visual_encoder.backbone" in name
                    or "net.goal_visual_encoder.backbone" in name
                ):
                    visual_encoder_params.append(param)
                else:
                    other_params.append(param)

        self.optimizer = optim.AdamW(
            [
                {"params": visual_encoder_params, "lr": encoder_lr},
                {"params": other_params, "lr": lr},
            ],
            lr=lr,
            weight_decay=wd,
            eps=eps,
        )


class MDDPPO(DecentralizedDistributedMixin, MPPO):
    pass
