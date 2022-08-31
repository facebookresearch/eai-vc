import torch
import torch.nn as nn
from higher.optim import DifferentiableOptimizer
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchrl.trainers import BatchSubSampler
from torchrl.envs.utils import step_tensordict
from torchrl.objectives import PPOLoss, ClipPPOLoss, GAE, TDEstimate


class DifferentiablePPO(nn.Module):
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
        num_envs: int,
        num_steps: int,
    ):
        super().__init__()
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_mini_batch = num_mini_batch
        self.num_epochs = num_epochs
        self.value_loss_coef = value_loss_coef
        self.num_envs = num_envs
        self.num_steps = num_steps

    def update(
        self,
        policy,
        storage,
        logger,
        opt: DifferentiableOptimizer,
    ):
        if self.use_gae:
            advantage_module = GAE(
                gamma=self.gamma,
                lmbda=self.gae_lambda,
                value_network=policy.critic,
                average_rewards=True,
            )
        else:
            advantage_module = TDEstimate(
                gamma=self.gamma, value_network=policy.critic, average_rewards=True
            )

        if self.use_clipped_value_loss:
            loss_module = ClipPPOLoss(
                policy.actor,
                policy.critic,
                advantage_module=advantage_module,
                entropy_coef=self.entropy_coef,
                critic_coef=self.value_loss_coef,
                gamma=self.gamma,
                clip_epsilon=self.clip_param,
                make_stateless=False,
            )
        else:
            loss_module = PPOLoss(
                policy.actor,
                policy.critic,
                advantage_module=advantage_module,
                entropy_coef=self.entropy_coef,
                critic_coef=self.value_loss_coef,
                gamma=self.gamma,
                make_stateless=False,
            )

        data_gen = BatchSubSampler(
            batch_size=self.num_envs * self.num_steps // self.num_mini_batch
        )
        batch_size = storage.batch_size
        storage_view = storage.view(batch_size[0] * batch_size[1])

        for _ in range(self.num_epochs):
            data = data_gen(storage_view)
            loss = loss_module(data.to_tensordict())
            loss_objective, loss_critic, loss_entropy = (
                loss["loss_objective"],
                loss["loss_critic"],
                loss.get("loss_entropy", 0),
            )
            total_loss = loss_objective + loss_critic + loss_entropy

            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            opt.step(total_loss)

            logger.collect_info("value_loss", loss_critic.mean().item())
            logger.collect_info("action_loss", loss_objective.mean().item())
            logger.collect_info("dist_entropy", loss_entropy.mean().item())
