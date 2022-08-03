#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from collections import defaultdict


class RolloutStorage:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states = torch.zeros(
            1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, 1)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)
        self.episode_step_index = [1] * (num_steps + 1)

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        actions,
        rewards,
        masks,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = self.step + 1
    
    def update_running_episode_step(self, masks):
        for i in range(self.num_envs):
            self.episode_step_index[i] += 1
            if masks[self.step][i].item() == 0:
                self.episode_step_index[i] = 1

    def after_update(self, rnn_hidden_states):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            rnn_hidden_states.detach()
        )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.step = 0

    def get_next_actions(self):
        next_action_observations = self.observations["demonstration"][self.step]
        actions = next_action_observations.clone()
        return actions

    def recurrent_generator(self, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            masks_batch = []
            index_batch = []

            for offset in range(num_envs_per_batch):
                # ind = perm[start_ind + offset]
                ind = start_ind + offset
                # Ignore OOB index
                if ind >= num_processes:
                    continue

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                index_batch.append(ind)

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                masks_batch,
                index_batch
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])
