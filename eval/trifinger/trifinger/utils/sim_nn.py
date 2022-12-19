"""
Run MoveCubePolicy to generate cube re-posititioning demos
"""

import sys
import os
import argparse
import numpy as np
import torch
import json

import utils.train_utils as t_utils
import utils.data_utils as d_utils
from trifinger_envs.cube_env import SimCubeEnv, ActionType
from trifinger_simulation.trifinger_platform import ObjectType
from utils.preprocess_trajs import MODEL_NAMES, get_model_and_transform
from utils.encoder_model import EncoderModel
from utils.execute_nn_policy import ExecuteNNPolicy

"""
Class to execute sequence of actions
"""


class SimNN:
    def __init__(
        self,
        policy_in_dim,
        max_a,
        state_type,
        downsample_time_step=0.2,
        traj_scale=1,
        goal_type=None,
        object_type="colored_cube",
        finger_type="trifingerpro",
        min_a_per_dim=None,
        max_a_per_dim=None,
        enable_shadows=False,
        camera_view="default",
        arena_color="default",
        task="move_cube",
        n_fingers_to_move=3,
    ):

        self.sim_time_step = 0.004
        self.downsample_time_step = downsample_time_step
        self.traj_scale = traj_scale
        self.n_fingers_to_move = n_fingers_to_move

        if object_type == "colored_cube":
            self.object_type = ObjectType.COLORED_CUBE
        elif object_type == "green_cube":
            self.object_type = ObjectType.GREEN_CUBE
        else:
            raise NameError

        # Set fix_cube_base flag based on task
        if task == "move_cube":
            fix_cube_base = False
            episode_steps = 1000  # TODO hardcoded
        elif task == "reach_cube":
            fix_cube_base = True
            episode_steps = 500  # TODO hardcoded
        else:
            raise NameError

        self.env = SimCubeEnv(
            goal_pose=None,  # passing None to sample a random trajectory
            action_type=ActionType.TORQUE,
            visualization=False,
            no_collisions=False,
            enable_cameras=True,
            finger_type=finger_type,
            time_step=self.sim_time_step,
            camera_delay_steps=0,
            object_type=self.object_type,
            enable_shadows=enable_shadows,
            camera_view=camera_view,
            arena_color=arena_color,
            fix_cube_base=fix_cube_base,
        )

        self.policy = ExecuteNNPolicy(
            policy_in_dim,
            max_a,
            state_type,
            self.env.action_space,
            self.env.platform,
            time_step=self.sim_time_step,
            downsample_time_step=self.downsample_time_step,
            episode_steps=episode_steps,
            training_traj_scale=self.traj_scale,
            finger_type=finger_type,
            goal_type=goal_type,
            min_a_per_dim=min_a_per_dim,
            max_a_per_dim=max_a_per_dim,
            n_fingers_to_move=self.n_fingers_to_move,
        )

    def close(self):
        self.env.close()
        del self.policy

    def execute_policy(
        self,
        traj,
        policy_state_dict,
        cam_name="image_60",
        save_dir=None,
        encoder=None,
        epoch=-1,
    ):

        obj_init_pos = traj["o_pos_cur"][0, :] / self.traj_scale
        obj_init_ori = traj["o_ori_cur"][0, :]
        # obj_goal_pos = traj["o_pos_des"][0, :] / self.traj_scale
        # obj_goal_ori = traj["o_ori_des"][0, :]
        # Use final object position in demo as goal
        obj_goal_pos = traj["o_pos_cur"][-1, :] / self.traj_scale
        obj_goal_ori = traj["o_ori_cur"][-1, :]
        init_pose = {"position": obj_init_pos, "orientation": obj_init_ori}
        goal_pose = {"position": obj_goal_pos, "orientation": obj_goal_ori}
        qpos_init = traj["robot_pos"][0, :]

        observation = self.env.reset(
            goal_pose_dict=goal_pose,
            init_pose_dict=init_pose,
            init_robot_position=qpos_init,
        )

        self.policy.reset(observation, policy_state_dict, traj, encoder=encoder)
        observation_list = []
        is_done = False

        while not is_done:
            action = self.policy.predict(observation)
            observation, reward, episode_done, info = self.env.step(action)

            policy_observation = self.policy.get_observation()

            is_done = self.policy.done or episode_done

            full_observation = {**observation, **policy_observation}

            observation_list.append(full_observation)

        # Compute actions (ftpos and joint state deltas) across trajectory
        d_utils.add_actions_to_obs(observation_list)

        # Get traj_dict and downsample
        traj_dict_raw = d_utils.get_traj_dict_from_obs_list(
            observation_list, scale=self.traj_scale
        )
        traj_dict = d_utils.downsample_traj_dict(
            traj_dict_raw,
            new_time_step=self.downsample_time_step,
        )

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"obs_epoch_{epoch+1}.npz")
            np.savez_compressed(save_path, data=observation_list)
            print(f"Saved sim rollout to {save_path}")

            # Plot actions
            expert_actions = traj["delta_ftpos"]
            title = "Fingertip position deltas (epoch: {})".format(epoch)
            save_name = f"action_epoch_{epoch+1}.png"
            save_path = os.path.join(save_dir, save_name)

            d_utils.plot_traj(
                title,
                save_path,
                [
                    "x1",
                    "y1",
                    "z1",
                    "x2",
                    "y2",
                    "z2",
                    "x3",
                    "y3",
                    "z3",
                ],
                {
                    "pred": {
                        "y": traj_dict["delta_ftpos"][:-1],
                        "x": traj["t"][:-1],
                        "marker": "x",
                    },
                    "demo": {
                        "y": expert_actions[:-1],
                        "x": traj["t"][:-1],
                        "marker": ".",
                    },
                },
            )

        return traj_dict
