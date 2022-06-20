"""
Demo on how to run the simulation using the Gym environment
This demo creates a SimCubeTrajectoryEnv environment and runs one episode using
a dummy policy.
"""

import sys
import os
import os.path
import argparse
import numpy as np

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from envs.cube_env import SimCubeEnv, ActionType
from policies.move_cube_policy import MoveCubePolicy


DEBUG_EPISODE_LENGTH = 10000 

def main(args):
    env = SimCubeEnv(
        goal_pose=None,  # passing None to sample a random trajectory
        action_type=ActionType.TORQUE,
        visualization=args.visualize,
        no_collisions=args.no_collisions,
        difficulty=1,
    )

    is_done = False
    observation = env.reset()
    t = 0

    policy = MoveCubePolicy(env.action_space, env.platform)

    while not is_done:
        action = policy.predict(observation, t)
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]

        is_done = policy.done

    if args.log:
        log_dir = "logs"
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        filename = os.path.join(log_dir, "sim_log.npz")
        np.savez_compressed(filename, data=policy.get_sim_log())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize sim")
    parser.add_argument("--no_collisions", "-nc", action="store_true", help="Visualize sim")
    parser.add_argument("--log", "-l", action="store_true", help="Save sim log")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
