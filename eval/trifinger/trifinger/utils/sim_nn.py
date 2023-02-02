"""
Run MoveCubePolicy to generate cube re-posititioning demos
"""

import sys
import os
import argparse
import numpy as np
import torch
import json

import utils.data_utils as d_utils
import utils.train_utils as t_utils
from trifinger_simulation.trifinger_platform import ObjectType
from utils.encoder_model import EncoderModel
from trifinger_envs.cube_env import ActionType
from trifinger_envs.gym_cube_env import MoveCubeEnv
from trifinger_envs.cube_reach import CubeReachEnv
from utils.policy import DeterministicPolicy
from utils.preprocess_trajs import MODEL_NAMES

"""
Class to execute sequence of actions
"""


class SimNN:
    def __init__(
        self,
        policy_in_dim,
        max_a,
        state_type,
        obj_state_type,
        downsample_time_step=0.2,
        traj_scale=1,
        goal_type=None,
        object_type="colored_cube",
        finger_type="trifingerpro",
        enable_shadows=False,
        camera_view="default",
        arena_color="default",
        task="move_cube",
        n_fingers_to_move=3,
    ):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.sim_time_step = 0.004
        self.downsample_time_step = downsample_time_step
        self.traj_scale = traj_scale
        self.n_fingers_to_move = n_fingers_to_move
        self.a_dim = self.n_fingers_to_move * 3
        self.task = task
        self.state_type = state_type
        self.obj_state_type = obj_state_type
        self.goal_type = goal_type

        step_size = int(self.downsample_time_step / self.sim_time_step)

        if object_type == "colored_cube":
            self.object_type = ObjectType.COLORED_CUBE
        elif object_type == "green_cube":
            self.object_type = ObjectType.GREEN_CUBE
        else:
            raise NameError

        # Set env based on task
        if self.task == "move_cube":
            self.env = MoveCubeEnv(
                goal_pose=None,  # passing None to sample a random trajectory
                action_type=ActionType.TORQUE,
                step_size=step_size,
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
                visual_observation=True,
                run_rl_policy=False,
            )

        elif self.task == "reach_cube":
            self.env = CubeReachEnv(
                action_type=ActionType.TORQUE,
                step_size=step_size,
                visualization=False,
                enable_cameras=True,
                finger_type=finger_type,
                camera_delay_steps=0,
                time_step=self.sim_time_step,
                object_type=self.object_type,
                enable_shadows=enable_shadows,
                camera_view=camera_view,
                arena_color=arena_color,
                visual_observation=True,
                run_rl_policy=False,
            )
        else:
            raise NameError

        # Load policy network
        self.policy = self.load_policy(policy_in_dim, max_a)

    def close(self):
        self.env.close()
        del self.policy

    def reset(self, expert_demo_dict, policy_state_dict, encoder=None):
        # Update policy weights
        self.policy.load_state_dict(policy_state_dict)

        # Reset environment with init and goal positions, scaled from cm -> m
        obj_init_pos = expert_demo_dict["o_pos_cur"][0, :] / self.traj_scale
        obj_init_ori = expert_demo_dict["o_ori_cur"][0, :]
        # Use final object position in demo as goal
        obj_goal_pos = expert_demo_dict["o_pos_cur"][-1, :] / self.traj_scale
        obj_goal_ori = expert_demo_dict["o_ori_cur"][-1, :]
        init_pose = {"position": obj_init_pos, "orientation": obj_init_ori}
        goal_pose = {"position": obj_goal_pos, "orientation": obj_goal_ori}
        qpos_init = expert_demo_dict["robot_pos"][0, :]

        if self.task == "move_cube":
            observation = self.env.reset(
                goal_pose_dict=goal_pose,
                init_pose_dict=init_pose,
                init_robot_position=qpos_init,
            )
        elif self.task == "reach_cube":
            observation = self.env.reset(
                init_pose_dict=init_pose,
                init_robot_position=qpos_init,
            )
        else:
            raise NameError

        # Set goals for bc state
        # Image goal from demo
        img_goal = expert_demo_dict["image_60"][-1]  # TODO hardcoded image_60
        if encoder is not None:
            self.o_goal = encoder.encode_img(img_goal).to(self.device)
        else:
            self.o_goal = torch.flatten(torch.FloatTensor(img_goal).to(self.device))

        # Object goal position, scaled to cm for policy
        self.o_goal_pos = (
            torch.FloatTensor(obj_goal_pos).to(self.device) * self.traj_scale
        )

        # Relative goal, scaled to cm for policy
        self.o_goal_pos_rel = (
            torch.FloatTensor(obj_goal_pos - obj_init_pos).to(self.device)
            * self.traj_scale
        )

        # print("O init: ", obj_init_pos)
        # print("O goal: ", self.o_goal_pos)
        # print("Rel goal: ", self.o_goal_pos_rel)

        return observation

    def execute_policy(
        self,
        expert_demo_dict,
        policy_state_dict,
        cam_name="image_60",
        save_dir=None,
        encoder=None,
        epoch=-1,
    ):
        # Reset env and update policy network
        observation_list = []
        observation = self.reset(expert_demo_dict, policy_state_dict, encoder=encoder)
        observation_list.append(observation)

        pred_actions = []
        episode_done = False
        action_counter = 0
        expert_actions = expert_demo_dict["delta_ftpos"]
        while not episode_done:
            # Get bc input tensor from observation
            # Scale observation by traj_scale, for bc policy
            q_cur = observation["robot_position"]
            ft_pos_cur = observation["ft_pos_cur"] * self.traj_scale
            # TODO hardcoded using image_60
            img = observation["camera_observation"]["camera60"]["image"]

            # TODO set o_state based on self.obj_state_type
            if self.obj_state_type in MODEL_NAMES:
                assert encoder is not None
                with torch.no_grad():
                    o_state = encoder.encode_img(img)
            elif self.obj_state_type == "pos":
                o_pos_cur = observation["object_position"] * self.traj_scale
                o_state = torch.FloatTensor(o_pos_cur).to(self.device)
            elif self.obj_state_type == "vertices":
                # TODO
                raise NotImplementedError
            elif self.obj_state_type == "rgb":
                o_state = torch.flatten(torch.FloatTensor(img).to(self.device))
            else:
                raise NameError

            # Make obs for policy
            ft_state = torch.FloatTensor(ft_pos_cur).to(self.device)
            obs_dict = {
                "ft_state": ft_state,
                "o_state": o_state,
                "o_goal": self.o_goal,
                "o_goal_pos": self.o_goal_pos,
                "o_goal_pos_rel": self.o_goal_pos_rel,
            }
            obs_tensor = t_utils.get_bc_obs_vec_from_obs_dict(
                obs_dict, self.state_type, self.goal_type
            )

            # Get action from policy, convert back to meters
            with torch.no_grad():
                a = self.policy(obs_tensor)
                a = self.policy.scale_to_range(a)
                a = self.policy.clip_action(a)
                pred_action = np.squeeze(a.cpu().detach().numpy()) / self.traj_scale

                three_finger_action = np.zeros(9)
                three_finger_action[: self.n_fingers_to_move * 3] = (
                    pred_action * self.traj_scale
                )
                pred_actions.append(three_finger_action)

            # TODO test env w/ groundtruth actions - this works
            # pred_action = expert_actions[action_counter, :] / self.traj_scale
            # pred_actions.append(pred_action)
            # action_counter += 1

            observation, reward, episode_done, info = self.env.step(pred_action)
            observation_list.append(observation)

        d_utils.add_actions_to_obs(observation_list)

        # Get traj_dict and downsample
        traj_dict = d_utils.get_traj_dict_from_obs_list(
            observation_list, scale=self.traj_scale
        )

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"obs_epoch_{epoch+1}.npz")
            np.savez_compressed(save_path, data=observation_list)
            print(f"Saved sim rollout to {save_path}")

            # Plot actions
            expert_actions = expert_demo_dict["delta_ftpos"]
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
                        "y": np.array(pred_actions),
                        "x": expert_demo_dict["t"][:-1],
                        "marker": "x",
                    },
                    "demo": {
                        "y": expert_actions[:-1],
                        "x": expert_demo_dict["t"][:-1],
                        "marker": ".",
                    },
                },
            )

        return traj_dict

    def load_policy(self, in_dim, max_a):
        # Load unscaled policy
        policy = DeterministicPolicy(
            in_dim=in_dim,
            out_dim=self.a_dim,
            max_a=max_a,
            device=self.device,
        )
        policy.eval()
        return policy
