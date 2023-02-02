import os
import sys
import numpy as np
import torch
import ipdb

import trifinger_simulation.finger_types_data

from control.impedance_controller import ImpedanceController
from control.custom_pinocchio_utils import CustomPinocchioUtils
import control.cube_utils as c_utils
from utils.policy import DeterministicPolicy
import utils.data_utils as d_utils
import utils.train_utils as t_utils
from utils.preprocess_trajs import MODEL_NAMES, get_model_and_transform


class ExecuteNNPolicy:
    """
    Follow fingertip position trajectory
    """

    def __init__(
        self,
        policy_in_dim,
        max_a,
        state_type,
        action_space,
        platform,
        finger_type="trifingerpro",
        goal_type=None,
        time_step=0.004,
        downsample_time_step=0.2,
        episode_steps=1000,
        training_traj_scale=1,
        min_a_per_dim=None,
        max_a_per_dim=None,
        n_fingers_to_move=3,
    ):
        """ """

        self.action_space = action_space
        self.time_step = time_step
        self.episode_steps = episode_steps
        self.downsample_time_step = downsample_time_step
        self.training_traj_scale = training_traj_scale
        self.state_type = state_type
        self.goal_type = goal_type
        self.n_fingers_to_move = n_fingers_to_move

        # TODO hardcoded
        robot_properties_path = (
            "../trifinger_simulation/trifinger_simulation/robot_properties_fingers"
        )

        urdf_file = trifinger_simulation.finger_types_data.get_finger_urdf(finger_type)

        finger_urdf_path = os.path.join(robot_properties_path, "urdf", urdf_file)

        # set platform (robot)
        self.platform = platform

        self.Nf = 3  # Number of fingers
        self.Nq = self.Nf * 3  # Number of joints in hand
        self.a_dim = self.n_fingers_to_move * 3

        # class with kinematics functions
        self.kinematics = CustomPinocchioUtils(
            self.platform.simfinger.finger_urdf_path,
            self.platform.simfinger.tip_link_names,
            self.platform.simfinger.link_names,
        )

        self.controller = ImpedanceController(self.kinematics, kp=[2000] * 9)

        self.traj_counter = 0

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.policy = self.load_policy(
            policy_in_dim,
            max_a,
            min_a_per_dim=min_a_per_dim,
            max_a_per_dim=max_a_per_dim,
        )

        self.encoder = None

    def reset(self, observation, policy_state_dict, expert_demo_dict, encoder=None):
        """
        args:
            expert_demo: for setting goal for bc policies
        """

        # mode and trajectory initializations
        self.traj_counter = 0

        self.set_traj_counter = 0

        # Initial ft pos and vel trajectories
        self.init_x = self.get_ft_pos(
            np.array(observation["robot_position"])
        )  # initial fingertip pos
        self.ft_pos_traj = np.expand_dims(self.init_x, 0)
        self.ft_vel_traj = np.zeros((1, 9))

        self.ft_pos_cur_plan = self.init_x.copy() * self.training_traj_scale

        self.t = 0

        self.done = False

        # Update policy weights
        self.policy.load_state_dict(policy_state_dict)

        # Update encoder weights
        if encoder:
            self.encoder = encoder

        # TODO hardcoded image_60
        img = expert_demo_dict["image_60"][-1]

        # this one applies the transform
        self.o_goal = self.encoder.encode_img(img).to(self.device)

        self.o_goal_pos = torch.FloatTensor(expert_demo_dict["o_pos_cur"][-1]).to(
            self.device
        )

        # Relative goal
        o_init_pos = (
            torch.FloatTensor(observation["object_position"]).to(self.device)
            * self.training_traj_scale
        )
        self.o_goal_pos_rel = self.o_goal_pos - o_init_pos
        # print("Q init: ", observation["robot_position"])
        # print("O init: ", o_init_pos)
        # print("O goal: ", self.o_goal_pos)
        # print("Rel goal: ", self.o_goal_pos_rel)

    def load_policy(self, in_dim, max_a, min_a_per_dim=None, max_a_per_dim=None):
        if min_a_per_dim is None or max_a_per_dim is None:
            # Load unscaled policy
            policy = DeterministicPolicy(
                in_dim=in_dim,
                out_dim=self.a_dim,
                max_a=max_a,
                device=self.device,
            )
        else:
            # Load scaled policy
            policy = ScaledDeterministicPolicy(
                in_dim=in_dim,
                out_dim=self.a_dim,
                max_a=max_a,
                device=self.device,
                min_a_per_dim=min_a_per_dim,
                max_a_per_dim=max_a_per_dim,
            )
        policy.eval()
        return policy

    def set_ft_traj(self, observation):
        # Run this every X timesteps (based on downsampling factor)

        # Scale observation by training_traj_scale, for bc policy
        o_pos_cur = observation["object_position"] * self.training_traj_scale
        q_cur = observation["robot_position"]
        ft_pos_cur = self.get_ft_pos(q_cur) * self.training_traj_scale

        # TODO: put this in a function?
        # Set object state based on obj_state_type
        # TODO hardcoded using image_60
        img = observation["camera_observation"]["camera60"]["image"]

        # this function applies the encoder transform
        o_state = self.encoder.encode_img(img)

        # d_utils.encode_img(
        #     self.encoder, self.encoder_transform, img
        # )

        # Make obs for policy based state_type
        o_state = o_state.to(self.device)
        ft_state = torch.FloatTensor(ft_pos_cur).to(self.device)
        if self.goal_type is not None:
            # TODO BC policy observation
            obs_dict = {
                "ft_state": ft_state,
                "o_state": o_state,
                "o_goal": self.o_goal,
                "o_goal_pos": self.o_goal_pos,
                "o_goal_pos_rel": self.o_goal_pos_rel,
            }
            obs = t_utils.get_bc_obs_vec_from_obs_dict(
                obs_dict, self.state_type, self.goal_type
            )
        else:
            # RL policy observation
            if self.state_type == "ftpos_obj":
                obs = torch.cat([ft_state, o_state])
            else:
                obs = o_state

        with torch.no_grad():
            a = self.policy(obs)
            a = self.policy.scale_to_range(a)
            a = self.policy.clip_action(a)

        pred_action = np.squeeze(a.cpu().detach().numpy())

        # Fill rest of actions with 0
        full_action = np.zeros(self.Nf * 3)
        full_action[: self.n_fingers_to_move * 3] = pred_action

        # Add ft delta to current ft pos
        ft_pos_next = self.ft_pos_cur_plan.copy() + full_action

        # Lin interp from current ft pos to next ft waypoint
        # Scale back to meters
        ft_traj = np.stack(
            (
                self.ft_pos_cur_plan / self.training_traj_scale,
                ft_pos_next / self.training_traj_scale,
            )
        )
        self.ft_pos_traj, self.ft_vel_traj = c_utils.lin_interp_pos_traj(
            ft_traj, self.downsample_time_step, self.time_step
        )

        self.ft_pos_traj = self.ft_pos_traj[1:]
        self.ft_vel_traj = self.ft_vel_traj[1:]

        # Reset traj counter
        self.traj_counter = 0

        self.set_traj_counter += 1

        # Update plan waypoint
        self.ft_pos_cur_plan = ft_pos_next.copy()

    def get_ft_des(self, observation):
        """Get fingertip desired pos"""

        ft_pos_des = self.ft_pos_traj[self.traj_counter, :]
        ft_vel_des = self.ft_vel_traj[self.traj_counter, :]

        if self.traj_counter < len(self.ft_pos_traj) - 1:
            self.traj_counter += 1

        return ft_pos_des, ft_vel_des

    def predict(self, observation):
        """
        Returns torques to command to robot
        """

        if self.t == 0 or self.traj_counter >= len(self.ft_pos_traj) - 1:
            self.set_ft_traj(observation)
            # print(self.set_traj_counter)

        # 3. Get current waypoints for finger tips
        x_des, dx_des = self.get_ft_des(observation)

        # 4. Get torques from controller
        q_cur = observation["robot_position"]
        dq_cur = observation["robot_velocity"]
        torque = self.controller.get_command_torque(x_des, dx_des, q_cur, dq_cur)

        self.t += 1

        if self.t >= self.episode_steps:
            self.done = True

        return self.clip_to_space(torque)

    def clip_to_space(self, action):
        """Clip action to action space"""

        return np.clip(action, self.action_space.low, self.action_space.high)

    def get_ft_pos(self, q):
        """Get fingertip positions given current joint configuration q"""

        ft_pos = np.array(self.kinematics.forward_kinematics(q)).reshape(self.Nq)
        return ft_pos

    def get_observation(self):
        obs = {
            "policy": {
                "controller": self.controller.get_observation(),
                "t": self.t,
            }
        }

        return obs
