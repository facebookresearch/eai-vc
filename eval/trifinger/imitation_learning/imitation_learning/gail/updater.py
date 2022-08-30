import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import call, instantiate
import imitation_learning
from imitation_learning.common.utils import (
    create_next_obs,
    extract_transition_batch,
    log_finished_rewards,
)
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rl_helper.common import DictDataset


class GAIL(nn.Module):
    def __init__(
        self,
        discriminator: DictConfig,
        policy_updater: DictConfig,
        # get_dataset_fn,
        dataset_path,
        batch_size: int,
        num_discrim_batches: int,
        discrim_opt: DictConfig,
        reward_update_freq: int,
        device,
        policy,
        num_envs,
        **kwargs,
    ):
        super().__init__()
        self.discriminator = instantiate(discriminator).to(device)
        self.policy_updater = instantiate(policy_updater, policy=policy)

        # self.dataset = call(get_dataset_fn)
        full_dataset_path = os.path.join(imitation_learning.__path__[0], dataset_path)

        self.dataset = DictDataset(
            torch.load(full_dataset_path),
            ["observations", "actions", "rewards", "terminals", "next_observations"],
        )

        self.expert_data = DataLoader(self.dataset, batch_size, shuffle=True)
        self.discrim_opt = instantiate(
            discrim_opt, params=self.discriminator.parameters()
        )
        self.reward_update_freq = reward_update_freq
        self._n_updates = 0
        self.batch_size = batch_size
        self.num_discrim_batches = num_discrim_batches

        self.device = device
        self._ep_rewards = torch.zeros(num_envs, device=self.device)

    def state_dict(self, **kwargs):
        return {
            **super().state_dict(**kwargs),
            "discrim_opt": self.discrim_opt.state_dict(),
        }

    def load_state_dict(self, state_dict, should_load_opt=False):
        opt_state = state_dict.pop("discrim_opt")
        # if should_load_opt:
        #     self.discrim_opt.load_state_dict(opt_state)
        return super().load_state_dict(state_dict)

    def viz_reward(self, cur_obs=None, action=None, next_obs=None) -> torch.Tensor:
        return self.discriminator.get_reward(
            cur_obs=cur_obs, actions=action, next_obs=next_obs, viz_reward=True
        )

    def _update_discriminator(self, policy, rollouts, logger):
        num_batches = len(rollouts) // self.batch_size
        agent_data = rollouts.data_generator(num_batches, get_next_obs=True)
        cur_num_batches = 0

        for expert_batch, agent_batch in zip(self.expert_data, agent_data):
            if (
                self.num_discrim_batches != -1
                and self.num_discrim_batches <= cur_num_batches
            ):
                break
            expert_d = self.discriminator(
                cur_obs=expert_batch["observations"],
                actions=expert_batch["actions"],
                next_obs=expert_batch["next_observations"],
                masks=(~expert_batch["terminals"].bool()).float(),
                policy=policy,
            )
            agent_d = self.discriminator(
                cur_obs=agent_batch["observation"],
                actions=agent_batch["action"],
                next_obs=agent_batch["next_obs"],
                masks=agent_batch["mask"],
                policy=policy,
            )

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d, torch.ones_like(expert_d, device=self.device)
            )
            agent_loss = F.binary_cross_entropy_with_logits(
                agent_d, torch.zeros_like(agent_d, device=self.device)
            )

            loss = expert_loss + agent_loss

            self.discrim_opt.zero_grad()
            loss.backward()
            self.discrim_opt.step()

            logger.collect_info("expert_loss", expert_loss.item())
            logger.collect_info("agent_loss", agent_loss.item())
            logger.collect_info("discim_loss", loss.item())
            cur_num_batches += 1

    def update(self, policy, rollouts, logger):
        if (
            self.reward_update_freq != -1
            and self._n_updates % self.reward_update_freq == 0
        ):
            self._update_discriminator(policy, rollouts, logger)

        obs, actions, next_obs, masks = extract_transition_batch(rollouts)
        with torch.no_grad():
            rollouts.rewards = self.discriminator.get_reward(
                cur_obs=obs,
                actions=actions,
                next_obs=next_obs,
                masks=masks,
                policy=policy,
            )
            self._ep_rewards = log_finished_rewards(rollouts, self._ep_rewards, logger)
        self.policy_updater.update(policy, rollouts, logger)
        self._n_updates += 1
