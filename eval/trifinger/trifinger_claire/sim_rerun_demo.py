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
sys.path.insert(0, os.path.join(base_path, ".."))

from envs.cube_env import SimCubeEnv, ActionType
from policies.execute_ftpos_delta_policy import ExecuteFtposDeltasPolicy
from policies.follow_ft_traj_policy import FollowFtTrajPolicy
import utils.data_utils as d_utils

SIM_TIME_STEP = 0.004

"""
Re-run demo; will save new demo-*.npz log in demo_dir/reruns/

Example command:
python sim_rerun_demo.py --log_path test/demos/difficulty-1/demo-0000.npz -v
"""


def main(args):
    env = SimCubeEnv(
        goal_pose=None,  # passing None to sample a random trajectory
        action_type=ActionType.TORQUE,
        visualization=args.visualize,
        no_collisions=args.no_collisions,
        enable_cameras=True,
        finger_type="trifingerpro",
        time_step=SIM_TIME_STEP,
        camera_delay_steps=0,
    )

    for i, demo_path in enumerate(args.log_paths):
        data = np.load(demo_path, allow_pickle=True)["data"]
        traj = d_utils.get_traj_dict_from_obs_list(data, scale=1)
        downsample_time_step = SIM_TIME_STEP
        # downsample_time_step = 0.2
        # traj = d_utils.downsample_traj_dict(traj, new_time_step=downsample_time_step)

        obj_init_pos = traj["o_pos_cur"][0, :]
        obj_init_ori = traj["o_ori_cur"][0, :]
        obj_goal_pos = traj["o_pos_des"][0, :]
        obj_goal_ori = traj["o_ori_des"][0, :]
        init_pose = {"position": obj_init_pos, "orientation": obj_init_ori}
        goal_pose = {"position": obj_goal_pos, "orientation": obj_goal_ori}
        qpos_init = traj["robot_pos"][0, :]
        observation = env.reset(
            goal_pose_dict=goal_pose,
            init_pose_dict=init_pose,
            init_robot_position=qpos_init,
        )

        ftpos_traj = traj["ft_pos_cur"]
        policy = FollowFtTrajPolicy(
            ftpos_traj,
            env.action_space,
            env.platform,
            time_step=SIM_TIME_STEP,
            downsample_time_step=downsample_time_step,
        )

        # TODO replay actions - don't use for now
        # Only works when replaying down-sampled actions (not actions at SIM_TIME_STEP freq)
        # ftpos_deltas = traj["delta_ftpos"]
        # policy = ExecuteFtposDeltasPolicy(ftpos_deltas, env.action_space, env.platform,
        #                                  time_step=SIM_TIME_STEP,
        #                                  downsample_time_step=downsample_time_step)

        policy.reset()
        observation_list = []
        is_done = False

        while not is_done:
            action = policy.predict(observation)
            observation, reward, episode_done, info = env.step(action)

            policy_observation = policy.get_observation()

            is_done = policy.done or episode_done

            full_observation = {**observation, **policy_observation}

            observation_list.append(full_observation)

        # Compute actions (ftpos and joint state deltas) across trajectory
        add_actions_to_obs(observation_list)
        demo_dir = os.path.split(demo_path)[0]
        demo_name = os.path.splitext(os.path.split(demo_path)[1])[0]
        log_dir = os.path.join(demo_dir, "reruns")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, f"{demo_name}.npz")
        np.savez_compressed(log_path, data=observation_list)
        print(f"Saved rerun of demo {demo_name} to {log_path}")


def add_actions_to_obs(observation_list):

    for t in range(len(observation_list) - 1):
        ftpos_cur = observation_list[t]["policy"]["controller"]["ft_pos_cur"]
        ftpos_next = observation_list[t + 1]["policy"]["controller"]["ft_pos_cur"]
        delta_ftpos = ftpos_next - ftpos_cur

        q_cur = observation_list[t]["robot_position"]
        q_next = observation_list[t + 1]["robot_position"]
        delta_q = q_next - q_cur

        action_dict = {"delta_ftpos": delta_ftpos, "delta_q": delta_q}
        observation_list[t]["action"] = action_dict

    action_dict = {
        "delta_ftpos": np.zeros(delta_ftpos.shape),
        "delta_q": np.zeros(delta_q.shape),
    }
    observation_list[-1]["action"] = action_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize sim")
    parser.add_argument(
        "--no_collisions", "-nc", action="store_true", help="Visualize sim"
    )
    parser.add_argument("--log_paths", "-l", nargs="*", type=str, help="Save sim log")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
