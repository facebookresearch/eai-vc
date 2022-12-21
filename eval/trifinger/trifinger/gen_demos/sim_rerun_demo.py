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

from trifinger_simulation.trifinger_platform import ObjectType
from trifinger_envs.cube_env import ActionType
from trifinger_envs.gym_cube_env import MoveCubeEnv
import utils.data_utils as d_utils


SIM_TIME_STEP = 0.004

"""
Re-run demo; will save new demo-*.npz log in demo_dir/reruns/

Example command:
python sim_rerun_demo.py --log_path test/demos/difficulty-1/demo-0000.npz -v
"""


def main(args):
    downsample_time_step = 0.4

    step_size = int(downsample_time_step / SIM_TIME_STEP)

    env = MoveCubeEnv(
        goal_pose=None,  # passing None to sample a random trajectory
        action_type=ActionType.TORQUE,
        step_size=step_size,
        visualization=args.visualize,
        no_collisions=args.no_collisions,
        enable_cameras=True,
        finger_type="trifinger_meta",
        time_step=SIM_TIME_STEP,
        camera_delay_steps=0,
        object_type=ObjectType.COLORED_CUBE,
        enable_shadows=True,
        camera_view="real",
        arena_color="real",
        visual_observation=True,
        run_rl_policy=False,
    )

    for i, demo_path in enumerate(args.log_paths):
        data = np.load(demo_path, allow_pickle=True)["data"]
        traj = d_utils.get_traj_dict_from_obs_list(data, scale=1)
        traj = d_utils.downsample_traj_dict(traj, new_time_step=downsample_time_step)

        obj_init_pos = traj["o_pos_cur"][0, :]
        obj_init_ori = traj["o_ori_cur"][0, :]
        obj_goal_pos = traj["o_pos_cur"][-1, :]
        obj_goal_ori = traj["o_ori_cur"][-1, :]
        init_pose = {"position": obj_init_pos, "orientation": obj_init_ori}
        goal_pose = {"position": obj_goal_pos, "orientation": obj_goal_ori}
        qpos_init = traj["robot_pos"][0, :]
        observation = env.reset(
            goal_pose_dict=goal_pose,
            init_pose_dict=init_pose,
            init_robot_position=qpos_init,
        )

        ftpos_deltas = traj["delta_ftpos"]

        observation_list = []
        action_counter = 0
        is_done = False

        while not is_done:
            action = ftpos_deltas[action_counter, :]

            observation, reward, episode_done, info = env.step(action)

            action_counter += 1
            observation_list.append(observation)

            is_done = episode_done

        print("Scaled success: ", observation_list[-1]["scaled_success"])
        final_pos_err = observation_list[-1]["achieved_goal_position_error"]
        print("Total episode length: ", len(observation_list))
        print("Final object position error: ", final_pos_err)

        # Compute actions (ftpos and joint state deltas) across trajectory
        d_utils.add_actions_to_obs(observation_list)
        demo_dir = os.path.split(demo_path)[0]
        demo_name = os.path.splitext(os.path.split(demo_path)[1])[0]
        log_dir = os.path.join(demo_dir, "reruns")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, f"{demo_name}.npz")
        np.savez_compressed(log_path, data=observation_list)
        print(f"Saved rerun of demo {demo_name} to {log_path}")


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
