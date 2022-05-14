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
        self.optimizer = optim.AdamW(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
            lr=lr,
            weight_decay=wd,
            eps=eps,
        )


class MDDPPO(DecentralizedDistributedMixin, MPPO):
    pass
