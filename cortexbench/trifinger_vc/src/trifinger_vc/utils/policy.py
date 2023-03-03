#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import copy

def construct_policy(task_state_type,train_ft_state_shape, pretrained_dim,task_goal_type,out_dim,max_a,device):
    o_state_dim = pretrained_dim

    if task_state_type == "obj":
        state_dim = o_state_dim
    elif task_state_type == "ftpos_obj":
        state_dim = o_state_dim + train_ft_state_shape
    elif task_state_type == "ftpos":
        state_dim = train_ft_state_shape
    else:
        raise NameError("Invalid state_type")

    if task_goal_type == "goal_none":
        in_dim = state_dim
    elif task_goal_type == "goal_cond":
        in_dim = state_dim + o_state_dim
    elif task_goal_type == "goal_o_pos":
        in_dim = state_dim + 3
    else:
        raise NameError("Invalid goal_type")

    policy = DeterministicPolicy(
        in_dim=in_dim,
        out_dim=out_dim,
        max_a=max_a,
        device=device,
    )

    return policy


class DeterministicPolicy(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        max_a=None,
        device="cpu",
    ):
        super().__init__()
        num_neurons = 2000
        self.activation = torch.nn.ReLU
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(in_dim, num_neurons),
            self.activation(),
            torch.nn.Linear(num_neurons, num_neurons),
            self.activation(),
            torch.nn.Linear(num_neurons, out_dim),
        )
        self.policy.to(device)
        self.device = device

        self.init_state = copy.deepcopy(self.policy.state_dict())

        self.max_a = max_a
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, state):
        action = self.policy(state)
        return action

    def reset(self):
        self.policy.load_state_dict(self.init_state)

    def clip_action(self, a):
        if self.max_a is not None:
            a = torch.where(
                a > self.max_a, torch.tensor([self.max_a]).to(self.device), a
            )
            a = torch.where(
                a < -self.max_a, -torch.tensor([self.max_a]).to(self.device), a
            )
        return a

    def scale_to_range(self, a):
        """Does not do anything; just returns a"""
        return a


class ScaledDeterministicPolicy(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        max_a=None,
        min_a_per_dim=None,
        max_a_per_dim=None,
        device="cpu",
    ):
        super().__init__()
        num_neurons = 2000
        self.activation = torch.nn.Tanh
        # self.activation = torch.nn.ReLU
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(in_dim, num_neurons),
            self.activation(),
            torch.nn.Linear(num_neurons, num_neurons),
            self.activation(),
            torch.nn.Linear(num_neurons, out_dim),
            torch.nn.Tanh(),
        )
        self.policy.to(device)
        self.device = device

        self.init_state = copy.deepcopy(self.policy.state_dict())

        self.max_a = max_a
        if min_a_per_dim is not None and max_a_per_dim is not None:
            assert max_a_per_dim >= min_a_per_dim
            self.min_a_per_dim = torch.unsqueeze(torch.Tensor(min_a_per_dim), 0).to(
                self.device
            )
            self.max_a_per_dim = torch.unsqueeze(torch.Tensor(max_a_per_dim), 0).to(
                self.device
            )
        else:
            self.min_a_per_dim = None
            self.max_a_per_dim = None
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, state):
        action = self.policy(state)
        return action

    def reset(self):
        self.policy.load_state_dict(self.init_state)

    def clip_action(self, a):
        if self.max_a is not None:
            a = torch.where(
                a > self.max_a, torch.tensor([self.max_a]).to(self.device), a
            )
            a = torch.where(
                a < -self.max_a, -torch.tensor([self.max_a]).to(self.device), a
            )
        if self.min_a_per_dim is not None and self.max_a_per_dim is not None:
            a = torch.where(a > self.max_a_per_dim, self.max_a_per_dim, a)
            a = torch.where(a < self.min_a_per_dim, self.min_a_per_dim, a)

        return a

    def scale_to_range(self, a):
        in_range_min = torch.ones(self.min_a_per_dim.shape).to(self.device) * -1
        in_range_max = torch.ones(self.min_a_per_dim.shape).to(self.device)

        a_scaled = t_utils.scale_to_range(
            a, in_range_min, in_range_max, self.min_a_per_dim, self.max_a_per_dim
        )
        return a_scaled
