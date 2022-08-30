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

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from envs.cube_env import SimCubeEnv, ActionType
from policies.execute_ftpos_delta_policy import ExecuteFtposDeltasPolicy
import utils.data_utils as d_utils

"""
Re-run demo; will save new demo-*.npz log in demo_dir/reruns/

Example command:
python sim_rerun_demo.py --log_path test/demos/difficulty-1/demo-0000.npz -v
"""

class SimMPC:
    def __init__(self, downsample_time_step=0.2, traj_scale=1):
        
        self.sim_time_step = 0.004
        self.downsample_time_step = downsample_time_step
        self.traj_scale=traj_scale

        self.env = SimCubeEnv(
            goal_pose=None,  # passing None to sample a random trajectory
            action_type=ActionType.TORQUE,
            visualization=False,
            no_collisions=False,
            enable_cameras=True,
            finger_type="trifingerpro",
            time_step=self.sim_time_step,
            camera_delay_steps=0,
        )

    def rollout_actions(self, traj, actions, save_path=None):

        obj_init_pos = traj["o_pos_cur"][0, :] / self.traj_scale
        obj_init_ori = traj["o_ori_cur"][0, :]
        obj_goal_pos = traj["o_pos_des"][0, :] / self.traj_scale
        obj_goal_ori = traj["o_ori_des"][0, :]
        init_pose = {"position": obj_init_pos, "orientation": obj_init_ori}
        goal_pose = {"position": obj_goal_pos, "orientation": obj_goal_ori}
        qpos_init = traj["robot_pos"][0, :]

        actions = actions / self.traj_scale

        observation = self.env.reset(goal_pose_dict=goal_pose, init_pose_dict=init_pose,
                                init_robot_position=qpos_init)

        policy = ExecuteFtposDeltasPolicy(actions, self.env.action_space, self.env.platform,
                                          time_step=self.sim_time_step,
                                          downsample_time_step=self.downsample_time_step)

        policy.reset()
        observation_list = []
        is_done = False

        while not is_done:
            action = policy.predict(observation)
            observation, reward, episode_done, info = self.env.step(action)

            policy_observation = policy.get_observation()

            is_done = policy.done or episode_done

            full_observation = {**observation, **policy_observation}

            observation_list.append(full_observation)

        # Compute actions (ftpos and joint state deltas) across trajectory
        add_actions_to_obs(observation_list)
        
        traj_dict = d_utils.get_traj_dict_from_obs_list(observation_list, scale=self.traj_scale)
        traj_dict = d_utils.downsample_traj_dict(traj_dict, new_time_step=self.downsample_time_step)

        if save_path is not None:
            np.savez_compressed(save_path, data=observation_list)
            print(f"Saved sim rollout to {save_path}")

        return traj_dict
        
def add_actions_to_obs(observation_list):

    for t in range(len(observation_list) - 1):
        ftpos_cur  = observation_list[t]["policy"]["controller"]["ft_pos_cur"]
        ftpos_next = observation_list[t+1]["policy"]["controller"]["ft_pos_cur"]
        delta_ftpos = ftpos_next - ftpos_cur

        q_cur  = observation_list[t]["robot_position"]
        q_next = observation_list[t+1]["robot_position"]
        delta_q = q_next - q_cur

        action_dict = {"delta_ftpos": delta_ftpos, "delta_q": delta_q}
        observation_list[t]["action"] = action_dict

    action_dict = {"delta_ftpos": np.zeros(delta_ftpos.shape), "delta_q": np.zeros(delta_q.shape)}
    observation_list[-1]["action"] = action_dict
