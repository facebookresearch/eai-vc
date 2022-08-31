from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from rl_utils.envs.pointmass.pointmass_env import PointMassEnv, PointMassParams
from rl_utils.envs.registry import full_env_registry


@dataclass(frozen=True)
class PointMassObstacleParams(PointMassParams):
    """
    :param square_obstacles: A list of obstacles where each obstacle is defined by a tuple with:
        * x,y position
        * Obstacle width
        * Obstacle length
        * Obstacle rotation angle (in degrees)
    """

    goal_thresh: float = 0.05
    square_obstacles: List[Tuple[Tuple[float, float], float, float, float]] = field(
        default_factory=list
    )


@full_env_registry.register_env("PointMassObstacle-v0")
class PointMassObstacleEnv(PointMassEnv):
    def __init__(
        self,
        num_envs: int,
        params: Optional[PointMassObstacleParams] = None,
        device: Optional[torch.device] = None,
        set_eval: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if params is None:
            params = PointMassObstacleParams()
        super().__init__(num_envs, params, device, set_eval, seed, **kwargs)
        self._circle_obs = []
        self._square_obs_T = []

        for ob_pos, x_len, y_len, rot in self._params.square_obstacles:
            rot = rot * (np.pi / 180.0)

            rot_T = torch.tensor(
                [
                    [np.cos(rot), -np.sin(rot), 0.0],
                    [np.sin(rot), np.cos(rot), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=self._device,
                dtype=torch.float,
            )
            trans_T = torch.tensor(
                [
                    [1.0, 0.0, ob_pos[0]],
                    [0.0, 1.0, ob_pos[1]],
                    [0.0, 0.0, 1.0],
                ],
                device=self._device,
                dtype=torch.float,
            )

            self._square_obs_T.append(
                (
                    trans_T @ rot_T,
                    x_len,
                    y_len,
                )
            )

    def _add_to_info(self, all_info):
        dists = self._get_dist()
        for i in range(self._batch_size):
            all_info[i]["at_goal"] = dists[i].item() < self._params.goal_thresh
        return all_info

    def forward(self, cur_pos, action):
        action = action.to(self._device)
        action = torch.clamp(action, -1.0, 1.0)
        new_pos = cur_pos + (action * self._params.dt)

        if self._params.clip_bounds:
            new_pos = torch.clamp(
                new_pos, -self._params.position_limit, self._params.position_limit
            )

        for ob_pos, ob_radius in self._circle_obs:
            local_pos = new_pos - ob_pos
            local_dist = torch.linalg.norm(local_pos, dim=-1)
            coll_idxs = torch.nonzero(local_dist < ob_radius)

            norm_pos = (local_pos / local_dist.view(-1, 1)) * ob_radius
            adjusted_pos = ob_pos + norm_pos
            new_pos[coll_idxs] = adjusted_pos[coll_idxs]

        inside_obstacle = self.is_inside_obstacle(new_pos)

        new_pos[inside_obstacle] = cur_pos[inside_obstacle]

        return new_pos

    def is_inside_obstacle(self, pos: torch.Tensor) -> torch.BoolTensor:
        """
        :param pos: A tensor of shape (N, 2).
        :returns: Tensor of shape (N,) indicating if the points were inside the obstacle.
        """

        homo_pos = torch.cat(
            [pos, torch.ones(pos.shape[0], 1, device=self._device)], dim=-1
        )
        inside_any_box = torch.zeros(
            pos.shape[0], device=self._device, dtype=torch.bool
        )
        for obs_T, xlen, ylen in self._square_obs_T:
            local_pos = torch.linalg.inv(obs_T) @ homo_pos.T

            inside_x = torch.logical_and(local_pos[0] < xlen, local_pos[0] > -xlen)
            inside_y = torch.logical_and(local_pos[1] < ylen, local_pos[1] > -ylen)
            inside_box = torch.logical_and(inside_x, inside_y)
            inside_any_box |= inside_box
        return inside_any_box
