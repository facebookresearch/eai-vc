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


def test_single_traj(train_or_test, traj_num, env, conf, ckpt_path, demo_info, exp_dir):
    """
    args:
        train_or_test: "test_demos" or "train_demos"
        traj_num: number of demo traj to test
        env: PyBullet env
        conf: configuration dict
        ckpt_path: ckpt path
        demo_info: full info for all demos
    """

    if "scale" in demo_info:
        training_traj_scale = demo_info["scale"]
    else:
        training_traj_scale = 1

    expert_demo = demo_info[f"{train_or_test}_demos"][traj_num]

    obj_init_pos = expert_demo["o_pos_cur"][0, :] / training_traj_scale
    obj_init_ori = expert_demo["o_ori_cur"][0, :] / training_traj_scale
    obj_goal_pos = expert_demo["o_pos_des"][0, :] / training_traj_scale
    obj_goal_ori = expert_demo["o_ori_des"][0, :] / training_traj_scale
    init_pose = {"position": obj_init_pos, "orientation": obj_init_ori}
    goal_pose = {"position": obj_goal_pos, "orientation": obj_goal_ori}

    downsample_time_step = demo_info["downsample_time_step"]
    demo_stats = demo_info[f"{train_or_test}_demo_stats"][traj_num]
    demo_stats["n_train_traj"] = len(demo_info["train_demos"]) # Number of training trajectories for policy
    demo_t = expert_demo["t"] # Trajectory timestamps

    print(f"Testing {train_or_test} traj {traj_num}")

    #algo = conf["algo"]["name"]
    #if "bc_obs_type" in conf["algo"]: bc_obs_type = conf["algo"]["bc_obs_type"]

    algo = conf.algo
    bc_obs_type = conf.bc_obs_type

    observation = env.reset(goal_pose_dict=goal_pose, init_pose_dict=init_pose)
    if algo == "mbirl":
        # Get checkpoint info
        ckpt_info = torch.load(ckpt_path)

        demo_stats["pred_traj"] = ckpt_info[f"{train_or_test}_pred_traj_per_demo"][traj_num].detach().numpy() / training_traj_scale
        demo_stats["pred_traj_t"] = demo_t
        ftpos_traj = ckpt_info[f"{train_or_test}_pred_traj_per_demo"][traj_num].detach().numpy()[:, :9] / training_traj_scale
        #ftpos_traj = expert_demo["ft_pos_cur"] # Use expert actions [FOR DEBUGGING]
        policy = FollowFtTrajPolicy(ftpos_traj, env.action_space, env.platform, time_step=SIM_TIME_STEP,
                                    downsample_time_step=downsample_time_step)
    elif algo == "bc":
        policy = BCPolicy(ckpt_path, expert_demo, bc_obs_type,
                          env.action_space, env.platform, time_step=SIM_TIME_STEP,
                          downsample_time_step=downsample_time_step, training_traj_scale=training_traj_scale)
    else:
        raise ValueError("Invalid arg")

    policy.reset()

    observation_list = []

    is_done = False
    while not is_done:
        action = policy.predict(observation)
        observation, reward, episode_done, info = env.step(action)

        policy_observation = policy.get_observation()

        is_done = policy.done or episode_done
    
        full_observation = {**observation, **policy_observation}

        if args.log_paths is not None: observation_list.append(full_observation)

    ### SAVE SIM LOG ###
    # Compute actions (ftpos and joint state deltas) across trajectory
    add_actions_to_obs(observation_list) 
    log_dir = os.path.join(exp_dir, "eval", train_or_test)
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f"traj_{traj_num}_log.npz")
    np.savez_compressed(log_path, data=observation_list, demo_data=[demo_stats])
    print(f"Saved {train_or_test} traj {traj_num} log to {log_path}")

    final_obj_pos_err = observation_list[-1]["achieved_goal"]["position_error"]
    return final_obj_pos_err

def main(args):
    env = SimCubeEnv(
        goal_pose=None,  # passing None to sample a random trajectory
        action_type=ActionType.TORQUE,
        visualization=args.visualize,
        no_collisions=args.no_collisions,
        enable_cameras=(not args.disable_cameras),
        finger_type="trifingerpro",
        time_step=SIM_TIME_STEP,
        camera_delay_steps=0,
    )
      
    for i, ckpt_path in enumerate(args.log_paths):

        print(f"Checkpoint {i}")

        exp_dir = os.path.split(os.path.split(ckpt_path)[0])[0]
        demo_info_path = os.path.join(exp_dir, "demo_info.pth")
        conf_path = os.path.join(exp_dir, "conf.pth")

        conf = torch.load(conf_path)

        # Load demo_info.pth and get object initial and goal pose, and test demo stats
        demo_info = torch.load(demo_info_path)
    
        type_to_eval = ["test"]
        if args.eval_train_and_test: type_to_eval.append("train")

        for train_or_test in type_to_eval:
            n_demos = len(demo_info[f"{train_or_test}_demos"])
            for traj_num in range(n_demos):
                final_obj_pos_err = test_single_traj(train_or_test, traj_num, env, conf, ckpt_path, demo_info, exp_dir)


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
    parser.add_argument("--disable_cameras", "-dc", action="store_true", help="Visualize goal")
    parser.add_argument("--no_collisions", "-nc", action="store_true", help="Visualize sim")
    parser.add_argument("--log_paths", "-l", nargs="*", type=str, help="Save sim log")
    parser.add_argument("--eval_train_and_test", action="store_true", help="Eval both train and test demos")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
