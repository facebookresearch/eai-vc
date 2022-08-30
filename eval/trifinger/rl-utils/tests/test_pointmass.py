import torch

from rl_utils.envs import create_vectorized_envs
from rl_utils.envs.pointmass import PointMassObstacleParams, PointMassParams


def test_create():
    envs = create_vectorized_envs(
        "PointMass-v0",
        32,
    )
    envs.reset()
    for _ in range(100):
        rnd_ac = torch.tensor(envs.action_space.sample())
        rnd_ac = rnd_ac.view(1, -1).repeat(32, 1)
        envs.step(rnd_ac)


def test_examples():
    envs = create_vectorized_envs("PointMass-v0", num_envs=32)

    envs = create_vectorized_envs(
        "PointMass-v0", num_envs=32, params=PointMassParams(dt=0.1, ep_horizon=10)
    )

    envs = create_vectorized_envs("PointMassObstacle-v0", num_envs=32)
    envs.reset()
    for _ in range(100):
        rnd_ac = torch.tensor(envs.action_space.sample())
        rnd_ac = rnd_ac.view(1, -1).repeat(32, 1)
        envs.step(rnd_ac)

    envs = create_vectorized_envs(
        "PointMassObstacle-v0",
        num_envs=32,
        params=PointMassObstacleParams(
            dt=0.1, ep_horizon=10, square_obstacles=[([0.5, 0.5], 0.11, 0.5, 45.0)]
        ),
    )
    envs.reset()
    for _ in range(100):
        rnd_ac = torch.tensor(envs.action_space.sample())
        rnd_ac = rnd_ac.view(1, -1).repeat(32, 1)
        envs.step(rnd_ac)
