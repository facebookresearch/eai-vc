"""
Load and execute predicted fingertip traj from log.pth file
"""

import sys
import os
import os.path
import argparse
import numpy as np
import torch

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from envs.cube_env import SimCubeEnv, ActionType
from policies.follow_ft_traj_policy import FollowFtTrajPolicy
from policies.bc_policy import BCPolicy

SIM_TIME_STEP = 0.004

def main(args):
    env = SimCubeEnv(
        goal_pose=None,  # passing None to sample a random trajectory
        action_type=ActionType.TORQUE,
        visualization=args.visualize,
        no_collisions=args.no_collisions,
        enable_cameras=True,
        finger_type="trifingerpro",
        time_step=SIM_TIME_STEP,
    )
      
    if args.log_paths:
        num_episodes = len(args.log_paths)
    else:
        num_episodes = 6

    ckpt_path = args.log_paths[0]
    exp_dir = os.path.split(os.path.split(ckpt_path)[0])[0]
    demo_info_path = os.path.join(exp_dir, "demo_info.pth")
    conf_path = os.path.join(exp_dir, "conf.pth")

    # Load algo (either "bc" or "mbirl")
    conf = torch.load(conf_path)
    algo = conf.algo

    # Load demo_info.pth and get object initial and goal pose
    TEST_TRAJ_NUM = 0
    demo_info = torch.load(demo_info_path)
    expert_demo = demo_info["test_demos"][TEST_TRAJ_NUM]
    obj_init_pos = expert_demo["o_cur_pos"][0, :]
    obj_init_ori = expert_demo["o_cur_ori"][0, :]
    obj_goal_pos = expert_demo["o_des_pos"][0, :]
    obj_goal_ori = expert_demo["o_des_ori"][0, :]
    expert_actions = expert_demo["delta_ftpos"]
    init_pose = {"position": obj_init_pos, "orientation": obj_init_ori}
    goal_pose = {"position": obj_goal_pos, "orientation": obj_goal_ori}
    downsample_time_step = demo_info["downsample_time_step"]

    # Get checkpoint info
    ckpt_info = torch.load(ckpt_path)

    for i in range(num_episodes):
        print(f"Running episode {i}")

        is_done = False

        observation = env.reset(goal_pose_dict=goal_pose, init_pose_dict=init_pose)
        if algo == "mbirl":
            ftpos_traj = ckpt_info["test_pred_traj_per_demo"][TEST_TRAJ_NUM].detach().numpy()
            #ftpos_traj = expert_demo["ft_pos_cur"]
            policy = FollowFtTrajPolicy(ftpos_traj, env.action_space, env.platform, time_step=SIM_TIME_STEP,
                                        downsample_time_step=downsample_time_step)
        elif algo == "bc":
            policy = BCPolicy(ckpt_path, expert_actions, env.action_space, env.platform, time_step=SIM_TIME_STEP,
                                        downsample_time_step=downsample_time_step)
        else:
            raise ValueError("Invalid arg")

        policy.reset()

        observation_list = []

        while not is_done:
            action = policy.predict(observation)
            observation, reward, episode_done, info = env.step(action)

            policy_observation = policy.get_observation()

            is_done = policy.done or episode_done
        
            full_observation = {**observation, **policy_observation}

            if args.log_paths is not None: observation_list.append(full_observation)

        if args.log_paths is not None:
            # Compute actions (ftpos and joint state deltas) across trajectory
            add_actions_to_obs(observation_list) 
            log_path = os.path.join(os.path.split(args.log_paths[i])[0], "pred_traj_log.npz")
            np.savez_compressed(log_path, data=observation_list)
            print(f"Saved episode {i} to {log_path}")

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
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize sim")
    parser.add_argument("--no_collisions", "-nc", action="store_true", help="Visualize sim")
    parser.add_argument("--log_paths", "-l", nargs="*", type=str, help="Save sim log")
    parser.add_argument("--algo", "-a", type=str, choices=["mbirl", "bc"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
