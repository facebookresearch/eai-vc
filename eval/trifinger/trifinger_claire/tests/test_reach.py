import gym
from torchrl.envs import GymWrapper
import numpy as np
from trifinger_simulation import trifingerpro_limits

import sys
import os

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, ".."))

import envs


def load_env():
    reach_env = gym.make("ReachEnv-v0")
    r = reach_env.reset()
    assert isinstance(r, dict)
    # envs.reach_env.FIRST_DEFAULT_GOAL
    assert (envs.reach_env.FIRST_DEFAULT_GOAL == r["desired_goal"][:3]).all()
    assert (envs.reach_env.SECOND_DEFAULT_GOAL == r["desired_goal"][3:6]).all()
    assert (envs.reach_env.THIRD_DEFAULT_GOAL == r["desired_goal"][6:9]).all()


def test_reward():
    reach_env = gym.make("ReachEnv-v0")
    r = reach_env.reset()
    import pdb

    for i in range(5):
        # action = trifingerpro_limits.robot_torque.low
        action = np.zeros(9)
        obs, reward, done, info = reach_env.step(action)
        assert np.abs(reward - 6.5) < 0.5

    # super hacky
    reach_env.env.env.env.env.goal = obs["observation"]
    obs, reward, done, info = reach_env.step(np.zeros(9))
    print(reward)
    # assert(np.abs(reward-10) < 0.5)


def new_goals():
    print("not implemented")
    # expect goal to be randomly sampled from valid parts (different each time), if fixed_goal = False
    # action move towards goal
    # reward to be 10 when goal is met


def move_to_goal():
    # DEFAULT_GOAL = np.array([ 0.102,0.141,0.095,0.102,0.141,0.095,0.102,0.141,0.095])
    DEFAULT_GOAL = np.array(
        [
            0.10295916,
            0.14167858,
            0.08190478,
            0.10295916,
            0.14167858,
            0.08190478,
            0.10295916,
            0.14167858,
            0.08190478,
        ]
    )

    reach_env = gym.make("ReachEnv-v0", render_mode="human")
    obs = reach_env.reset()
    # joint pos to ftip pos
    pos = obs["observation"]
    old_reward = 0
    for i in range(10):

        delta = (DEFAULT_GOAL - pos) * 100
        obs, reward, done, info = reach_env.step(delta)
        pos = obs["observation"]
        print(reward)
        close_to_goal = np.abs(reward - 8.2) < 0.5
        assert reward >= old_reward or close_to_goal

        old_reward = reward


# load_env()
# test_reward()
move_to_goal()
