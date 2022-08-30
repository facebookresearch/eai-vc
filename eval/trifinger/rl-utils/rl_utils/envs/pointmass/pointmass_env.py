from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from gym import spaces
from torch.distributions import Uniform

from rl_utils.envs.registry import full_env_registry
from rl_utils.envs.vec_env.vec_env import FINAL_OBS_KEY, VecEnv


@dataclass(frozen=True)
class PointMassParams:
    """
    :param force_eval_start_dist: Generate the start positions from the eval offset.
    :param force_train_start_dist: Generate the start positions from the train offset.
    :param clip_bounds: Clip the agent to be within [-position_limit, position_limit]^2 ?
    :param clip_actions: Clip the actions to be within -1 to 1.
    :param ep_horizon: The length of the episode.
    :param custom_reward: A function that takes as input the current position,
        previous position, and action  and outputs a reward value. All are PyTorch
        tensors of shape (N,) where N is the number of environments.
    """

    force_eval_start_dist: bool = False
    force_train_start_dist: bool = True
    clip_bounds: bool = True
    clip_actions: bool = True
    ep_horizon: int = 5
    num_train_regions: int = 4
    start_state_noise: float = np.pi / 20
    dt: float = 0.2
    reward_dist_pen: float = 1 / 10.0
    start_idx: int = -1
    radius: float = 1.0
    eval_offset: float = 0.0
    train_offset: float = np.pi / 4
    position_limit: float = 1.5
    transition_noise: float = 0.0
    random_start_region_sample: bool = True
    custom_reward: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None


@full_env_registry.register_env("PointMass-v0")
class PointMassEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        params: Optional[PointMassParams] = None,
        device: Optional[torch.device] = None,
        set_eval: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if params is None:
            params = PointMassParams()
        if device is None:
            device = torch.device("cpu")
        self._batch_size = num_envs
        self._params = params

        self._device = device
        self._goal = torch.tensor([0.0, 0.0]).to(self._device)
        self._ep_step = 0
        self._prev_dist_idx = -1

        self._ep_rewards = []
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), seed=seed)
        ac_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), seed=seed)

        self._is_eval = set_eval or self._params.force_eval_start_dist
        if self._params.force_train_start_dist:
            self._is_eval = False

        if self._is_eval:
            regions = self.get_regions(
                self._params.eval_offset, self._params.start_state_noise
            )
        else:
            regions = self.get_regions(
                self._params.train_offset, self._params.start_state_noise
            )

        if self._params.start_state_noise != 0:
            self._start_distributions = Uniform(regions[:, 0], regions[:, 1])
        else:
            self._start_distributions = SingleSampler(regions[:, 0])

        super().__init__(
            self._batch_size,
            obs_space,
            ac_space,
        )

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def forward(self, cur_pos, action):
        action = action.to(self._device)
        if self._params.clip_actions:
            action = torch.clamp(action, -1.0, 1.0)
        new_pos = cur_pos + (action * self._params.dt)
        if self._params.transition_noise != 0.0:
            action += self._params.transition_noise * torch.randn(
                action.shape, device=self._device
            )

        if self._params.clip_bounds:
            new_pos = torch.clamp(
                new_pos, -self._params.position_limit, self._params.position_limit
            )
        return new_pos

    def step(self, action):
        self.cur_pos = self.forward(self.cur_pos, action)
        self._ep_step += 1
        self._store_actions.append(action)

        is_done = self._ep_step >= self._params.ep_horizon
        reward = self._get_reward(action)
        self._ep_rewards.append(reward)

        all_is_done = torch.tensor(
            [is_done for _ in range(self._batch_size)], dtype=torch.bool
        )
        dist_to_goal = torch.linalg.norm(
            self._goal - self.cur_pos, dim=-1, keepdims=True
        )

        all_info = [
            {"dist_to_goal": dist_to_goal[i].item()} for i in range(self._batch_size)
        ]
        all_info = self._add_to_info(all_info)

        if is_done:
            store_actions = torch.stack(self._store_actions, dim=1)
            action_magnitudes = torch.linalg.norm(store_actions, dim=-1)
            final_obs = self._get_obs()
            for i in range(self._batch_size):
                all_info[i]["episode"] = {
                    "r": torch.stack(self._ep_rewards).sum(0)[i].item(),
                    "max_action_magnitude": action_magnitudes[i].max().item(),
                    "avg_action_magnitude": action_magnitudes[i].mean().item(),
                }
                all_info[i][FINAL_OBS_KEY] = final_obs[i]
            self.reset()

        return (self._get_obs(), reward, all_is_done, all_info)

    def get_images(self, mode=None, img_dim=64, **kwargs) -> np.ndarray:
        def convert_coordinate(coord):
            # Normalize position to [0,1]
            norm_pos = (coord + self._params.position_limit) / (
                2 * self._params.position_limit
            )

            # Convert position to image space
            return (norm_pos * img_dim).to(torch.long)

        def write_to_img(img, pos, size, color):
            lower_x = max(pos[0] - size, 0)
            upper_x = min(pos[0] + size, img_dim)

            lower_y = max(pos[1] - size, 0)
            upper_y = min(pos[1] + size, img_dim)

            img[lower_x:upper_x, lower_y:upper_y] = color
            return img

        agent_pos = convert_coordinate(self.cur_pos)
        goal_pos = convert_coordinate(self._goal)
        entity_size = img_dim // 32

        img = np.full((self._batch_size, img_dim, img_dim, 3), 255, dtype=np.uint8)
        agent_color = [8, 143, 143]
        goal_color = [224, 17, 95]

        for env_i in range(self._batch_size):
            img[env_i] = write_to_img(
                img[env_i], agent_pos[env_i], entity_size, agent_color
            )
            img[env_i] = write_to_img(img[env_i], goal_pos, entity_size, goal_color)

        return img

    def _add_to_info(self, all_info):
        return all_info

    def _get_dist(self):
        return torch.linalg.norm(self._goal - self.cur_pos, dim=-1, keepdims=True)

    def _get_reward(self, action):
        if self._params.custom_reward is None:
            dist_to_goal = torch.linalg.norm(
                self._goal - self.cur_pos, dim=-1, keepdims=True
            )

            reward = -self._params.reward_dist_pen * dist_to_goal
        else:
            reward = self._params.custom_reward(self.cur_pos, self._prev_pos, action)
        self._prev_pos = self.cur_pos.detach().clone()
        return reward  # noqa: R504

    def get_regions(self, offset, spread):
        inc = np.pi / 2

        centers = [offset + i * inc for i in range(4)]

        return torch.tensor(
            [[center - spread, center + spread] for center in centers]
        ).to(self._device)

    def _get_dist_idx(self, batch_size):
        if not self._params.random_start_region_sample:
            new_dist_idx = self._prev_dist_idx + 1
            new_dist_idx = new_dist_idx % self._params.num_train_regions
            self._prev_dist_idx = new_dist_idx
            return torch.full((batch_size,), new_dist_idx)

        if self._is_eval:
            return torch.randint(0, 4, (batch_size,))
        else:
            if self._params.start_idx == -1:
                return torch.randint(0, self._params.num_train_regions, (batch_size,))
            else:
                return torch.tensor([self._params.start_idx]).repeat(batch_size)

    def _sample_start(self, batch_size, offset_start):
        idx = self._get_dist_idx(batch_size).to(self._device)
        samples = self._start_distributions.sample(idx.shape)
        ang = samples.gather(1, idx.view(-1, 1)).view(-1)

        return (
            torch.stack(
                [
                    self._params.radius * torch.cos(ang),
                    self._params.radius * torch.sin(ang),
                ],
                dim=-1,
            ).to(self._device)
            + offset_start
        )

    def reset(self):
        self.cur_pos = self._sample_start(self._batch_size, self._goal)
        self._ep_step = 0
        self._ep_rewards = []
        self._prev_pos = self.cur_pos.detach().clone()
        self._store_actions = []

        return self._get_obs()

    def _get_obs(self):
        return self.cur_pos.clone()


class SingleSampler:
    def __init__(self, point):
        self.point = point

    def sample(self, shape):
        return self.point.unsqueeze(0).repeat(shape[0], 1)
