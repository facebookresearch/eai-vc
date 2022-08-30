"""
Run MoveCubePolicy to generate cube re-posititioning demos
"""

import sys
import os
import os.path
import argparse
import numpy as np
import pybullet
import random
import torch

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from envs.cube_env import SimCubeEnv, ActionType
from policies.execute_ftpos_delta_policy import ExecuteFtposDeltasPolicy

SIM_TIME_STEP = 0.004

def main(args):
    env = SimCubeEnv(
        goal_pose=None,  # passing None to sample a random trajectory
        action_type=ActionType.TORQUE,
        visualization=args.visualize,
        no_collisions=args.no_collisions,
        difficulty=args.difficulty,
        enable_cameras=True,
        finger_type="trifingerpro",
        time_step=SIM_TIME_STEP,
        camera_delay_steps=0,
    )

    if args.log_paths:
        num_episodes = len(args.log_paths)
    else:
        num_episodes = 6

    downsample_time_step=0.2
    ftpos_deltas=np.zeros((19,9))
    policy = ExecuteFtposDeltasPolicy(ftpos_deltas, env.action_space, env.platform, time_step=SIM_TIME_STEP,
                                downsample_time_step=downsample_time_step)

    #8[0-7]: zero actions init from waypoints 0-7 from demo
    #9[0-7]: random actions init from waypoints 0-7 from demo
    # Iterate through all train and test demos in traj_info?
    # num_episodes is length of log_paths

    traj_info = torch.load(args.seed_demos_path)
    training_traj_scale = traj_info["scale"]

    NUM_WAYPTS_TO_START_AT = 8

    i = 0
    for split_name in ["train", "test"]:
        for traj in traj_info[f"{split_name}_demos"]:
            if i >= num_episodes: break
            print(f"Running episode {i}")

            waypt = i % NUM_WAYPTS_TO_START_AT

            obj_init_pos = traj["o_pos_cur"][0, :] / training_traj_scale
            obj_init_ori = traj["o_ori_cur"][0, :]
            o_init_pose = {"position": obj_init_pos, "orientation": obj_init_ori}
            qpos_init = traj["robot_pos"][waypt, :]
            observation = env.reset(init_pose_dict=o_init_pose, init_robot_position=qpos_init)

            if args.difficulty == 8:
                ftpos_deltas=np.zeros((19,9))
            elif args.difficulty == 9:
                ftpos_deltas=np.random.rand(19,9) * 0.02  - 0.01 # [-0.01, 0.01] meters
            else:
                raise ValueError("Invalid difficulty")

            policy.reset(ftpos_deltas=ftpos_deltas)

            observation_list = []

            x, y = 0,0
            is_done = False
            while not is_done:
                action = policy.predict(observation)
                observation, reward, episode_done, info = env.step(action)

                policy_observation = policy.get_observation()

                ## TODO for testing - manually move cube
                #obj_id = env.platform.cube._object_id
                #x = random.uniform(-0.1, 0.1)
                #y = random.uniform(-0.1, 0.1)
                #y += 0.0001
                #x += 0.0001
                #pybullet.resetBasePositionAndOrientation(obj_id, posObj=[x,y,0.0325], ornObj=[0,0,0,1])

                is_done = policy.done or episode_done

                full_observation = {**observation, **policy_observation}

                if args.log_paths is not None: observation_list.append(full_observation)

            if args.log_paths is not None:
                # Compute actions (ftpos and joint state deltas) across trajectory
                add_actions_to_obs(observation_list)
                log_path = args.log_paths[i]
                np.savez_compressed(log_path, data=observation_list)
                print(f"Saved episode {i} to {log_path}")

            i += 1

def add_actions_to_obs(observation_list):

    for t in range(len(observation_list) - 1):
        ftpos_cur  = observation_list[t]["policy"]["controller"]["ft_pos_cur"]
        ftpos_next = observation_list[t+1]["policy"]["controller"]["ft_pos_cur"]
        delta_ftpos = ftpos_next - ftpos_cur

        q_cur  = observation_list[t]["robot_observation"]["position"]
        q_next = observation_list[t+1]["robot_observation"]["position"]
        delta_q = q_next - q_cur

        action_dict = {"delta_ftpos": delta_ftpos, "delta_q": delta_q}
        observation_list[t]["action"] = action_dict

    action_dict = {"delta_ftpos": np.zeros(delta_ftpos.shape), "delta_q": np.zeros(delta_q.shape)}
    observation_list[-1]["action"] = action_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", "-d", type=int, choices=[8,9], help="Difficulty level", default=8)
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize sim")
    parser.add_argument("--no_collisions", "-nc", action="store_true", help="Visualize sim")
    parser.add_argument("--log_paths", "-l", nargs="*", type=str, help="Save sim log")
    parser.add_argument("--seed_demos_path", "-s", type=str, help="Path to preloaded_demos.pth")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
