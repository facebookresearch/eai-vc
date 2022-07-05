from typing import Dict, Optional, Tuple

import torch
from gym import spaces
from habitat import logger
from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net, Policy
from torch import nn as nn

from eai.sensors import ImageGoalRotationSensor
from eai.transforms import get_transform
from eai.utils import load_encoder
from eai.visual_encoder import VisualEncoder


class EAINet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        backbone: str,
        resnet_baseplanes: int,
        vit_use_fc_norm: bool,
        vit_global_pool: bool,
        vit_use_cls: bool,
        vit_mask_ratio: float,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
        use_augmentations: bool,
        use_augmentations_test_time: bool,
        randomize_augmentations_over_envs: bool,
        pretrained_encoder: Optional[str],
        freeze_backbone: bool,
        run_type: str,
        avgpooled_image: bool,
        augmentations_name: str,
        drop_path_rate: float,
    ):
        super().__init__()

        rnn_input_size = 0

        # visual encoder
        assert "rgb" in observation_space.spaces

        name = "resize"
        if use_augmentations and run_type == "train":
            name = augmentations_name
        if use_augmentations_test_time and run_type == "eval":
            name = augmentations_name
        self.visual_transform = get_transform(name, size=128)
        self.visual_transform.randomize_environments = randomize_augmentations_over_envs

        self.visual_encoder = VisualEncoder(
            image_size=128,
            backbone=backbone,
            input_channels=3,
            resnet_baseplanes=resnet_baseplanes,
            resnet_ngroups=resnet_baseplanes // 2,
            vit_use_fc_norm=vit_use_fc_norm,
            vit_global_pool=vit_global_pool,
            vit_use_cls=vit_use_cls,
            vit_mask_ratio=vit_mask_ratio,
            avgpooled_image=avgpooled_image,
            drop_path_rate=drop_path_rate,
        )

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.visual_encoder.output_size, hidden_size),
            nn.ReLU(True),
        )

        rnn_input_size += hidden_size

        # goal embedding
        if ImageGoalRotationSensor.cls_uuid in observation_space.spaces:
            name = "resize"
            if use_augmentations and run_type == "train":
                name = augmentations_name
            if use_augmentations_test_time and run_type == "eval":
                name = augmentations_name
            self.goal_transform = get_transform(name, size=128)
            self.goal_transform.randomize_environments = (
                randomize_augmentations_over_envs
            )

            self.goal_visual_encoder = VisualEncoder(
                image_size=128,
                backbone=backbone,
                input_channels=3,
                resnet_baseplanes=resnet_baseplanes,
                resnet_ngroups=resnet_baseplanes // 2,
                vit_use_fc_norm=vit_use_fc_norm,
                vit_global_pool=vit_global_pool,
                vit_use_cls=vit_use_cls,
                vit_mask_ratio=vit_mask_ratio,
                avgpooled_image=avgpooled_image,
                drop_path_rate=drop_path_rate,
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
        if ImageGoalRotationSensor.cls_uuid in observations:
            goal = observations[ImageGoalRotationSensor.cls_uuid]
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
        resnet_baseplanes: int = 32,
        vit_use_fc_norm: bool = False,
        vit_global_pool: bool = False,
        vit_use_cls: bool = False,
        vit_mask_ratio: Optional[float] = None,
        hidden_size: int = 512,
        rnn_type: str = "GRU",
        num_recurrent_layers: int = 1,
        use_augmentations: bool = False,
        use_augmentations_test_time: bool = False,
        randomize_augmentations_over_envs: bool = False,
        pretrained_encoder: Optional[str] = None,
        freeze_backbone: bool = False,
        run_type: str = "train",
        avgpooled_image: bool = False,
        augmentations_name: str = "",
        drop_path_rate: float = 0.0,
        **kwargs
    ):
        super().__init__(
            EAINet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                vit_use_fc_norm=vit_use_fc_norm,
                vit_global_pool=vit_global_pool,
                vit_use_cls=vit_use_cls,
                vit_mask_ratio=vit_mask_ratio,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
                use_augmentations=use_augmentations,
                use_augmentations_test_time=use_augmentations_test_time,
                randomize_augmentations_over_envs=randomize_augmentations_over_envs,
                pretrained_encoder=pretrained_encoder,
                freeze_backbone=freeze_backbone,
                run_type=run_type,
                avgpooled_image=avgpooled_image,
                augmentations_name=augmentations_name,
                drop_path_rate=drop_path_rate,
            ),
            dim_actions=action_space.n,  # for action distribution
        )

    @classmethod
    def from_config(cls, config: Config, observation_space: spaces.Dict, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            backbone=config.RL.POLICY.backbone,
            resnet_baseplanes=config.RL.POLICY.resnet_baseplanes,
            vit_use_fc_norm=config.RL.POLICY.vit_use_fc_norm,
            vit_global_pool=config.RL.POLICY.vit_global_pool,
            vit_use_cls=config.RL.POLICY.vit_use_cls,
            vit_mask_ratio=config.RL.POLICY.vit_mask_ratio,
            hidden_size=config.RL.POLICY.hidden_size,
            rnn_type=config.RL.POLICY.rnn_type,
            num_recurrent_layers=config.RL.POLICY.num_recurrent_layers,
            use_augmentations=config.RL.POLICY.use_augmentations,
            use_augmentations_test_time=config.RL.POLICY.use_augmentations_test_time,
            randomize_augmentations_over_envs=config.RL.POLICY.randomize_augmentations_over_envs,
            pretrained_encoder=config.RL.POLICY.pretrained_encoder,
            freeze_backbone=config.RL.POLICY.freeze_backbone,
            run_type=config.RUN_TYPE,
            avgpooled_image=config.RL.POLICY.avgpooled_image,
            augmentations_name=config.RL.POLICY.augmentations_name,
            drop_path_rate=config.RL.POLICY.drop_path_rate,
        )
