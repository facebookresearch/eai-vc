import gym
import numpy as np
import pdb

import os
import sys

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))
import envs


# Claire: I set step_size to 1 (was set to 1000, which was doing 1000 control steps for 1 sim step)
# For now, I think stick to step_size=1, but I will think about if we want to change this in the future for training policies
env = gym.make("ReachEnv-v0", render_mode='human', step_size=1)
prev_obs = env.reset()
delta = env.action_space.high - env.action_space.low
for i in range(10):
    #action = (np.random.random(9)*2) -1

    # Claire: Some test actions (
    #action = np.zeros(9) # Hold fingers at initial positions
    action = np.array([0,0,0.001]*3) # Move fingers up by 1mm every sim step

    obs,reward,done,info = env.step(action)

    error = np.abs((prev_obs["achieved_goal"] - obs["achieved_goal"]) - action)
    prev_obs = obs
    #pdb.set_trace()
