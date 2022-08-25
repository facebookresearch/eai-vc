import numpy as np
import gym
import torch

from causal_world.envs import CausalWorld
from causal_world.task_generators.reaching import ReachingTaskGenerator
from imitation_learning.common.trifinger_pushing_task import PushingTaskGenerator


class CausalWorldReacherWrapper(CausalWorld):
    def __init__(self, start_state_noise=0.0, skip_frame=10, max_ep_horizon=100):
        default_start_joint_pos = np.array(
            [
                0.0,
                0.78539816,
                -0.78539816,
                0.0,
                0.78539816,
                -0.78539816,
                0.0,
                0.78539816,
                -0.78539816,
            ]
        )
        start_noise = np.random.randn(9) * start_state_noise
        start_joint_pos = default_start_joint_pos + start_noise
        task = ReachingTaskGenerator(joint_positions=start_joint_pos)
        super().__init__(
            task=task,
            enable_visualization=False,
            skip_frame=skip_frame,
            max_episode_length=max_ep_horizon,
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(18,), dtype=float
        )

    def step(self, action):
        obs, reward, done, infos = super(CausalWorldReacherWrapper, self).step(action)
        joint_pos = obs[1:10]
        # convert to cm
        end_effector_pos = obs[-18:-9]  # *100.0
        return (
            torch.Tensor(np.hstack([joint_pos, end_effector_pos])),
            reward,
            done,
            infos,
        )

    def reset(self):
        obs = super(CausalWorldReacherWrapper, self).reset()
        joint_pos = obs[1:10]
        # convert to cm
        end_effector_pos = obs[-18:-9]  # *100.0
        return torch.Tensor(np.hstack([joint_pos, end_effector_pos]))


class CausalWorldPushingWrapper(CausalWorld):
    def __init__(
        self, starting_joint_state=None, skip_frame=10, max_episode_length=100
    ):
        task = PushingTaskGenerator(joint_positions=starting_joint_state)
        super().__init__(
            task=task, skip_frame=skip_frame, max_episode_length=max_episode_length
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(21,), dtype=float
        )

    def step(self, action):
        obs, reward, done, infos = super(CausalWorldPushingWrapper, self).step(action)
        joint_pos = obs[1:10]
        # convert to cm
        end_effector_pos = obs[19:28]
        block_pos = obs[49:52]
        return np.hstack([joint_pos, end_effector_pos, block_pos]), reward, done, infos

    def reset(self):
        # observation dimension
        # time left 1
        # joint positions 9
        # joint vels 9
        # ee pos 9
        # tool block type 1
        # block size
        # block cart pos 3
        # block orient 4
        # block lin vel 3
        # block ang vel 3
        # goal block type 1
        # goal block size 3
        # goal cart pos 3
        # goal orient 4

        obs = super(CausalWorldPushingWrapper, self).reset()
        joint_pos = obs[1:10]
        # convert to cm
        end_effector_pos = obs[19:28]
        block_pos = obs[49:52]
        return np.hstack([joint_pos, end_effector_pos, block_pos])
