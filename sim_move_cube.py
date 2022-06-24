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


def main(args):
    env = SimCubeEnv(
        goal_pose=None,  # passing None to sample a random trajectory
        action_type=ActionType.TORQUE,
        visualization=args.visualize,
        no_collisions=args.no_collisions,
        difficulty=args.difficulty,
        enable_cameras=True,
        finger_type="trifingerpro",
    )

    policy = MoveCubePolicy(env.action_space, env.platform)
      
    if args.log_paths:
        num_episodes = len(args.log_paths)
    else:
        num_episodes = 6

    t = 0
    for i in range(num_episodes):
        print(f"Running episode {i}")

        is_done = False
        env.reset()
        policy.reset()

        observation_list = []
        observation = env._create_observation(t, env._initial_action)

        while not is_done:
            action = policy.predict(observation)
            observation, reward, episode_done, info = env.step(action)
            t = info["time_index"]

            policy_observation = policy.get_observation()

            is_done = policy.done or episode_done
        
            full_observation = {**observation, **policy_observation}

            if args.log_paths is not None: observation_list.append(full_observation)

        if args.log_paths is not None:
            log_path = args.log_paths[i]
            np.savez_compressed(log_path, data=observation_list)
            print(f"Saved episode {i} to {log_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", "-d", type=int, choices=[1,2,3], help="Difficulty level", default=1)
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize sim")
    parser.add_argument("--no_collisions", "-nc", action="store_true", help="Visualize sim")
    parser.add_argument("--log_paths", "-l", nargs="*", type=str, help="Save sim log")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
