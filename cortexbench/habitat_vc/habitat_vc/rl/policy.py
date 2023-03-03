#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
from gym import spaces
from habitat.config import Config
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net, Policy
from torch import nn as nn

from habitat_vc.rl.imagenav.sensors import ImageGoalRotationSensor
from habitat_vc.visual_encoder import VisualEncoder
from habitat_vc.models.freeze_batchnorm import convert_frozen_batchnorm


class EAINet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        input_image_size,
        backbone_config,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
        use_augmentations: bool,
        use_augmentations_test_time: bool,
        run_type: str,
        freeze_backbone: bool,
        freeze_batchnorm: bool,
        global_pool: bool,
        use_cls: bool,
    ):
        super().__init__()

        rnn_input_size = 0

        # visual encoder
        assert "rgb" in observation_space.spaces

        if (use_augmentations and run_type == "train") or (
            use_augmentations_test_time and run_type == "eval"
        ):
            use_augmentations = True

        self.visual_encoder = VisualEncoder(
            backbone_config=backbone_config,
            image_size=input_image_size,
            global_pool=global_pool,
            use_cls=use_cls,
            use_augmentations=use_augmentations,
        )

        self.visual_fc = nn.Sequential(
            nn.Linear(self.visual_encoder.output_size, hidden_size),
            nn.ReLU(True),
        )

        rnn_input_size += hidden_size

        # object goal embedding
        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]) + 1
            )
            self.obj_categories_embedding = nn.Embedding(self._n_object_categories, 32)
            rnn_input_size += 32

        # image goal embedding
        if ImageGoalRotationSensor.cls_uuid in observation_space.spaces:
            self.goal_visual_encoder = VisualEncoder(
                backbone_config=backbone_config,
                image_size=input_image_size,
                global_pool=global_pool,
                use_cls=use_cls,
                use_augmentations=use_augmentations,
                loaded_backbone_data=self.visual_encoder.get_loaded_backbone_data()
                if freeze_backbone
                else None,
            )

            self.goal_visual_fc = nn.Sequential(
                nn.Linear(self.goal_visual_encoder.output_size, hidden_size),
                nn.ReLU(True),
            )

            rnn_input_size += hidden_size

        # previous action embedding
        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        rnn_input_size += 32

        # state encoder
        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        # TODO: move this to the model files
        # freeze backbone
        if freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False
            has_goal_encoder = hasattr(self, "goal_visual_encoder")
            if has_goal_encoder:
                for p in self.goal_visual_encoder.backbone.parameters():
                    p.requires_grad = False
            if freeze_batchnorm:
                self.visual_encoder = convert_frozen_batchnorm(self.visual_encoder)
                if has_goal_encoder:
                    self.goal_visual_encoder = convert_frozen_batchnorm(
                        self.goal_visual_encoder
                    )

        # save configuration
        self._hidden_size = hidden_size

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def transform_images(self, observations, number_of_envs):
        images = observations["rgb"]

        imagenav_task = ImageGoalRotationSensor.cls_uuid in observations

        # concatenate images
        if imagenav_task:
            goal_images = observations[ImageGoalRotationSensor.cls_uuid]
            x = torch.cat([images, goal_images], dim=0)
        else:
            x = images

        x = (
            x.permute(0, 3, 1, 2).float() / 255
        )  # convert channels-last to channels-first
        x = self.visual_encoder.visual_transform(x, number_of_envs)

        return x.chunk(2, dim=0) if imagenav_task else x

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []

        # number of environments
        N = rnn_hidden_states.size(0)

        rgb, goal_rgb = self.transform_images(observations, N)

        # visual encoder
        rgb = self.visual_encoder(rgb)
        rgb = self.visual_fc(rgb)
        x.append(rgb)

        # goal embedding
        if ImageGoalRotationSensor.cls_uuid in observations:
            goal_rgb = self.goal_visual_encoder(goal_rgb)
            goal_rgb = self.goal_visual_fc(goal_rgb)
            x.append(goal_rgb)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        # previous action embedding
        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )
        x.append(prev_actions)

        # state encoder
        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(out, rnn_hidden_states, masks)

        return out, rnn_hidden_states


@baseline_registry.register_policy
class EAIPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        input_image_size,
        backbone_config,
        hidden_size: int = 512,
        rnn_type: str = "GRU",
        num_recurrent_layers: int = 1,
        use_augmentations: bool = False,
        use_augmentations_test_time: bool = False,
        run_type: str = "train",
        freeze_backbone: bool = False,
        freeze_batchnorm: bool = False,
        global_pool: bool = False,
        use_cls: bool = False,
        **kwargs
    ):
        super().__init__(
            EAINet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                input_image_size=input_image_size,
                backbone_config=backbone_config,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
                use_augmentations=use_augmentations,
                use_augmentations_test_time=use_augmentations_test_time,
                run_type=run_type,
                freeze_backbone=freeze_backbone,
                freeze_batchnorm=freeze_batchnorm,
                global_pool=global_pool,
                use_cls=use_cls,
            ),
            dim_actions=action_space.n,  # for action distribution
        )

    @classmethod
    def from_config(cls, config: Config, observation_space: spaces.Dict, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            input_image_size=config.RL.POLICY.input_image_size,
            backbone_config=config.model,
            hidden_size=config.RL.POLICY.hidden_size,
            rnn_type=config.RL.POLICY.rnn_type,
            num_recurrent_layers=config.RL.POLICY.num_recurrent_layers,
            use_augmentations=config.RL.POLICY.use_augmentations,
            use_augmentations_test_time=config.RL.POLICY.use_augmentations_test_time,
            run_type=config.RUN_TYPE,
            freeze_backbone=config.RL.POLICY.freeze_backbone,
            freeze_batchnorm=config.RL.POLICY.freeze_batchnorm,
            global_pool=config.RL.POLICY.global_pool,
            use_cls=config.RL.POLICY.use_cls,
        )
