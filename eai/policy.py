from typing import Dict, Optional, Tuple

import torch
from gym import spaces
from habitat import logger
from habitat.config import Config
from habitat.tasks.nav.nav import ImageGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net, Policy
from torch import nn as nn

from eai.transforms import get_transform
from eai.utils import load_encoder
from eai.visual_encoder import VisualEncoder


class EAINet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        backbone: str,
        baseplanes: int,
        mask_ratio: float,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
        use_augmentations: bool,
        use_augmentations_test_time: bool,
        randomize_augmentations_over_envs: bool,
        pretrained_encoder: Optional[str],
        freeze_backbone: bool,
        run_type: str,
    ):
        super().__init__()

        rnn_input_size = 0

        # visual encoder
        assert "rgb" in observation_space.spaces

        name = "resize"
        if use_augmentations and run_type == "train":
            name = "shift+jitter"
        if use_augmentations_test_time and run_type == "eval":
            name = "shift+jitter"
        self.visual_transform = get_transform(name, size=128)
        self.visual_transform.randomize_environments = randomize_augmentations_over_envs

        self.visual_encoder = VisualEncoder(
            image_size=128,
            backbone=backbone,
            input_channels=3,
            baseplanes=baseplanes,
            ngroups=baseplanes // 2,
            mask_ratio=mask_ratio,
        )

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.visual_encoder.output_size, hidden_size),
            nn.ReLU(True),
        )

        rnn_input_size += hidden_size

        # goal embedding
        if ImageGoalSensor.cls_uuid in observation_space.spaces:
            name = "resize"
            if use_augmentations and run_type == "train":
                name = "shift+jitter"
            if use_augmentations_test_time and run_type == "eval":
                name = "shift+jitter"
            self.goal_transform = get_transform(name, size=128)
            self.goal_transform.randomize_environments = (
                randomize_augmentations_over_envs
            )

            self.goal_visual_encoder = VisualEncoder(
                image_size=128,
                backbone=backbone,
                input_channels=3,
                baseplanes=baseplanes,
                ngroups=baseplanes // 2,
                mask_ratio=mask_ratio,
            )

            self.goal_visual_fc = nn.Sequential(
                nn.Flatten(),
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

        # pretrained weights
        if pretrained_encoder is not None:
            msg = load_encoder(self.visual_encoder, pretrained_encoder)
            logger.info("Using weights from {}: {}".format(pretrained_encoder, msg))
            msg = load_encoder(self.goal_visual_encoder, pretrained_encoder)
            logger.info("Using weights from {}: {}".format(pretrained_encoder, msg))

        # freeze backbone
        if freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False
            for p in self.goal_visual_encoder.backbone.parameters():
                p.requires_grad = False

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

        # visual encoder
        rgb = observations["rgb"]
        rgb = self.visual_transform(rgb, N)
        rgb = self.visual_encoder(rgb)
        rgb = self.visual_fc(rgb)
        x.append(rgb)

        # goal embedding
        if ImageGoalSensor.cls_uuid in observations:
            goal = observations[ImageGoalSensor.cls_uuid]
            goal = self.goal_transform(goal, N)
            goal = self.goal_visual_encoder(goal)
            goal = self.goal_visual_fc(goal)
            x.append(goal)

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
        backbone: str = "resnet18",
        baseplanes: int = 32,
        mask_ratio: float = 0.5,
        hidden_size: int = 512,
        rnn_type: str = "GRU",
        num_recurrent_layers: int = 1,
        use_augmentations: bool = False,
        use_augmentations_test_time: bool = False,
        randomize_augmentations_over_envs: bool = False,
        pretrained_encoder: Optional[str] = None,
        freeze_backbone: bool = False,
        run_type: str = "train",
        **kwargs
    ):
        super().__init__(
            EAINet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                backbone=backbone,
                baseplanes=baseplanes,
                mask_ratio=mask_ratio,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
                use_augmentations=use_augmentations,
                use_augmentations_test_time=use_augmentations_test_time,
                randomize_augmentations_over_envs=randomize_augmentations_over_envs,
                pretrained_encoder=pretrained_encoder,
                freeze_backbone=freeze_backbone,
                run_type=run_type,
            ),
            dim_actions=action_space.n,  # for action distribution
        )

    @classmethod
    def from_config(cls, config: Config, observation_space: spaces.Dict, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            backbone=config.RL.POLICY.backbone,
            baseplanes=config.RL.POLICY.baseplanes,
            mask_ratio=config.RL.POLICY.mask_ratio,
            hidden_size=config.RL.POLICY.hidden_size,
            rnn_type=config.RL.POLICY.rnn_type,
            num_recurrent_layers=config.RL.POLICY.num_recurrent_layers,
            use_augmentations=config.RL.POLICY.use_augmentations,
            use_augmentations_test_time=config.RL.POLICY.use_augmentations_test_time,
            randomize_augmentations_over_envs=config.RL.POLICY.randomize_augmentations_over_envs,
            pretrained_encoder=config.RL.POLICY.pretrained_encoder,
            freeze_backbone=config.RL.POLICY.freeze_backbone,
            run_type=config.RUN_TYPE,
        )
