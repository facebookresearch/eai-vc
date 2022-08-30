from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_utils.models import build_rnn_state_encoder

from imitation_learning.utils.distributions import FixedNormal, FixedCategorical
from torchrl.modules import (
    ConvNet,
    TensorDictModule,
    TensorDictSequence,
    ProbabilisticActor,
    ValueOperator,
    ActorCriticOperator,
)
from torchrl.data import TensorDict
from rl_utils.common import set_seed


def init_weights(m, gain=1):
    if isinstance(m, nn.Linear):

        torch.nn.init.orthogonal_(m.weight, gain=gain)
        m.bias.data.fill_(0.0)


class CategoricalParams(nn.Module):
    """Returns the parameters for a categorical distribution."""

    def __init__(self, hidden_size, action_dim):
        super().__init__()

        self.linear = nn.Linear(hidden_size, action_dim)
        self.apply(partial(init_weights, gain=0.01))

    def forward(self, x):
        x = self.linear(x)
        return x


class DiagGaussianParams(nn.Module):
    """Returns the parameters for a normal distribution."""

    def __init__(self, hidden_size, action_dim, std_init, squash_mean):
        super().__init__()

        if squash_mean:
            self.fc_mean = nn.Linear(hidden_size, action_dim)
        else:
            self.fc_mean = nn.Sequential(
                nn.Linear(hidden_size, action_dim),
                nn.ReLU(),
            )
        self.logstd = nn.Parameter(torch.full((1, action_dim), float(std_init)))
        self.apply(init_weights)

    def forward(self, x):
        action_mean = self.fc_mean(x)

        action_logstd = self.logstd.expand_as(action_mean)
        return action_mean, action_logstd.exp()


class QNet(nn.Module):
    """Q-value network"""

    def __init__(self, action_dim, obs_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, observation, action):
        input = torch.cat([observation, action], dim=-1)
        out = self.fc1(input)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


class Policy(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        action_is_discrete,
        hidden_size,
        recurrent_hidden_size,
        is_recurrent,
        num_envs,
        std_init=0.0,
        squash_mean=False,
    ):

        super().__init__()

        self.num_envs = num_envs
        self.hidden_size = hidden_size
        self.recurrent_hidden_size = recurrent_hidden_size
        self.action_is_discrete = action_is_discrete

        if isinstance(obs_shape, dict):
            is_visual_obs = any([len(v) == 3 for k, v in obs_shape.items()])
        else:
            is_visual_obs = len(obs_shape) == 3

        if is_visual_obs:
            conv_subnet = ConvNet(
                depth=3,
                num_cells=[32, 64, 32],
                kernel_sizes=[(8, 8), (4, 4), (3, 3)],
                strides=[4, 2, 1],
                activation_class=nn.ReLU,
            )
            conv_net = nn.Sequential(
                conv_subnet,
                nn.Linear(32 * obs_shape[0] * obs_shape[1], hidden_size),
                nn.ReLU(),
            )

            self.backbone = TensorDictModule(
                module=conv_net, in_keys=["observation"], out_keys=["hidden"]
            )

            input_size = hidden_size
        else:
            net = nn.Sequential()
            self.backbone = TensorDictModule(
                module=net, in_keys=["observation"], out_keys=["hidden"]
            )
            input_size = obs_shape[0]

        # TODO: Wrap in torch rl LSTM and add to self.module

        if is_recurrent:
            self.rnn_encoder = (
                build_rnn_state_encoder(  # keep for later stage, supported by torch rl
                    recurrent_hidden_size, recurrent_hidden_size
                )
            )
        else:
            # Pass through
            rnn_net = nn.Identity()
            self.rnn_encoder = TensorDictModule(
                module=rnn_net, in_keys=["hidden"], out_keys=["hidden"]
            )

        if action_is_discrete:
            dist_params_net = CategoricalParams(hidden_size, action_dim)
            distribution_class = FixedCategorical
            param_keys = ["logits"]
        else:
            dist_params_net = DiagGaussianParams(
                hidden_size, action_dim, std_init, squash_mean
            )
            distribution_class = FixedNormal
            param_keys = ["loc", "scale"]

        actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            dist_params_net,
        )
        # TODO: check if actor should also use hidden
        actor_module = TensorDictModule(
            module=actor, in_keys=["hidden"], out_keys=param_keys
        )
        self.target_entropy = -float(np.prod(action_dim))
        actor = ProbabilisticActor(
            module=actor_module,
            dist_param_keys=param_keys,
            distribution_class=distribution_class,
            default_interaction_mode="random",
        )

        qvalue_subnet = QNet(
            action_dim=action_dim, obs_size=input_size, hidden_size=hidden_size
        )
        # TOOD: check initiialization qnet and value net same
        value_subnet = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.qvalue_net = ValueOperator(
            qvalue_subnet, in_keys=["action", "observation"]
        )
        self.value_net = ValueOperator(value_subnet, in_keys=["observation"])

        self.apply(partial(init_weights, gain=np.sqrt(2)))

        self.hidden = TensorDictSequence(self.backbone, self.rnn_encoder)
        self.actor_critic = ActorCriticOperator(self.hidden, actor, self.qvalue_net)
        self.actor = self.actor_critic.get_policy_operator()

    # TODO: fix following functions
    def get_value(self, td):
        return self.actor_critic(td)["state_action_value"]

    def evaluate_actions(self, td):
        critic_td = self.actor_critic(td)
        critic_value = critic_td["state_action_value"]
        action = td["action"]

        dist, _ = self.actor.get_dist(td)
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return {
            "log_prob": action_log_prob,
            "value": critic_value,
            "dist_entropy": dist_entropy,
        }

    def forward(self, td):
        return self.hidden(td)

    def get_action_dist(self, td):
        self.forward(td)
        dist, _ = self.actor.get_dist(td)
        return dist

    def act(self, td, deterministic=False):
        critic_td = self.actor_critic(td)
        critic_value = critic_td["state_action_value"]
        dist, _ = self.actor.get_dist(td)

        if deterministic:
            action = dist.mode
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        if critic_td.get("recurrent_hidden_states", None) is None:
            td["recurrent_hidden_states"] = torch.zeros(
                self.num_envs, self.recurrent_hidden_size
            )

        td["action"] = action.long() if self.action_is_discrete else action
        td["sample_log_prob"] = action_log_probs
        td["value_preds"] = critic_value
        td["dist_entropy"] = dist_entropy
