import gym
import numpy as np
import pdb

import os
import sys

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))
import envs


env = gym.make("ReachEnv-v0", render_mode='human')
prev_obs = env.reset()
delta = env.action_space.high - env.action_space.low
for i in range(2000):
    action = (np.random.random(9)*2) -1
    obs,reward,done,info = env.step(action)

    error = np.abs((prev_obs["achieved_goal"] - obs["achieved_goal"]) - action)
    prev_obs = obs
    pdb.set_trace()
