#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

from torch import nn as nn
from habitat_baselines.utils.common import CategoricalNet


class ILPolicy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    def forward(self, *x):
        features, rnn_hidden_states = self.net(*x)
        distribution = self.action_distribution(features)
        distribution_entropy = distribution.entropy().mean()

        return distribution.logits, rnn_hidden_states, distribution_entropy

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=True,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        distribution_entropy = distribution.entropy().mean()

        return action, rnn_hidden_states, distribution_entropy

    def get_value(self, *x):
        raise NotImplementedError

    def evaluate_actions(self, *x):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass
