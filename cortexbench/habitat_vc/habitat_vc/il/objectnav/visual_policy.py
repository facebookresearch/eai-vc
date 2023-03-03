#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from gym import Space
from habitat import Config, logger
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoalSensor,
)
from habitat_baselines.rl.ppo import Net

from habitat_vc.il.objectnav.custom_baseline_registry import custom_baseline_registry
from habitat_vc.visual_encoder import VisualEncoder

from habitat_vc.il.objectnav.rnn_state_encoder import RNNStateEncoder
from habitat_vc.il.objectnav.policy import ILPolicy
from habitat_vc.models.freeze_batchnorm import convert_frozen_batchnorm


class ObjectNavILNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(
        self,
        observation_space: Space,
        model_config: Config,
        backbone_config: Config,
        num_actions: int,
        run_type: str,
    ):
        super().__init__()
        self.model_config = model_config
        rnn_input_size = 0

        rgb_config = model_config.RGB_ENCODER
        # Init the RGB visual encoder
        assert rgb_config.model_type in [
            "VisualEncoder",
            "None",
        ], "RGB_ENCODER.model_type must be 'VisualEncoder', or 'None'."

        use_augmentations = False
        if (rgb_config.use_augmentations and run_type == "train") or (
            rgb_config.use_augmentations_test_time and run_type == "eval"
        ):
            use_augmentations = True

        self.visual_encoder = VisualEncoder(
            image_size=rgb_config.image_size,
            backbone_config=backbone_config,
            global_pool=rgb_config.global_pool,
            use_cls=rgb_config.use_cls,
            use_augmentations=use_augmentations,
        )

        self.visual_fc = nn.Sequential(
            nn.Linear(self.visual_encoder.output_size, rgb_config.hidden_size),
            nn.ReLU(True),
        )

        rnn_input_size += rgb_config.hidden_size
        logger.info("RGB encoder is {}".format(rgb_config.model_type))

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[
                0
            ]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[0] == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(
                input_compass_dim, self.compass_embedding_dim
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]) + 1
            )
            logger.info("Object categories: {}".format(self._n_object_categories))
            self.obj_categories_embedding = nn.Embedding(self._n_object_categories, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.rnn_input_size = rnn_input_size

        # freeze backbone
        if rgb_config.freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False
            if rgb_config.freeze_batchnorm:
                self.visual_encoder = convert_frozen_batchnorm(self.visual_encoder)

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=model_config.STATE_ENCODER.num_recurrent_layers,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def transform_images(self, observations, number_of_envs):
        x = observations["rgb"]

        x = (
            x.permute(0, 3, 1, 2).float() / 255
        )  # convert channels-last to channels-first
        x = self.visual_encoder.visual_transform(x, number_of_envs)

        return x

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_obs = observations["rgb"]

        N = rnn_hidden_states.size(1)

        x = []

        if len(rgb_obs.size()) == 5:
            observations["rgb"] = rgb_obs.contiguous().view(
                -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
            )
        # visual encoder
        rgb = self.transform_images(observations, N)
        rgb = self.visual_encoder(rgb)
        rgb = self.visual_fc(rgb)
        x.append(rgb)

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))

        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(
                compass_observations.float().squeeze(dim=1)
            )
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(-1, object_goal.size(2))
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


@custom_baseline_registry.register_il_policy
class ObjectNavILPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        backbone_config: Config,
        model_config: Config,
        run_type: str,
    ):
        super().__init__(
            ObjectNavILNet(
                observation_space=observation_space,
                model_config=model_config,
                backbone_config=backbone_config,
                num_actions=action_space.n,
                run_type=run_type,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            backbone_config=config.model,
            model_config=config.MODEL,
            run_type=config.RUN_TYPE,
        )
