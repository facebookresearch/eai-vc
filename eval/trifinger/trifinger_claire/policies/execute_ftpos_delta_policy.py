import os
import sys
import numpy as np
import enum

import trifinger_simulation.finger_types_data
import trifinger_simulation.pinocchio_utils

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, ".."))

from control.impedance_controller import ImpedanceController
from control.custom_pinocchio_utils import CustomPinocchioUtils
import control.cube_utils as c_utils


class ExecuteFtposDeltasPolicy:
    """
    Execute fingertip position deltas
    """

    def __init__(
        self,
        ftpos_deltas,
        action_space,
        platform,
        time_step=0.001,
        downsample_time_step=0.001,
        episode_steps=1000,
    ):

        """
        ftpos_deltas: fingertip position delta sequence [T, 9]
        """

        self.action_space = action_space
        self.time_step = time_step
        self.downsample_time_step = downsample_time_step
        self.episode_steps = episode_steps

        self.ftpos_deltas = ftpos_deltas

        # TODO hardcoded
        robot_properties_path = (
            "../trifinger_simulation/trifinger_simulation/robot_properties_fingers"
        )

        urdf_file = trifinger_simulation.finger_types_data.get_finger_urdf(
            "trifingerpro"
        )

        finger_urdf_path = os.path.join(robot_properties_path, "urdf", urdf_file)

        # initial joint positions (lifting the fingers up)
        self.joint_positions = np.array(
            [-0.08, 1.15, -1.5] * 3
        )  # "down and out" position

        # set platform (robot)
        self.platform = platform

        self.Nf = 3  # Number of fingers
        self.Nq = self.Nf * 3  # Number of joints in hand

        # class with kinematics functions
        self.kinematics = CustomPinocchioUtils(
            self.platform.simfinger.finger_urdf_path,
            self.platform.simfinger.tip_link_names,
            self.platform.simfinger.link_names,
        )

        self.controller = ImpedanceController(self.kinematics, kp=[2000] * 9)

    def reset(self, ftpos_deltas=None):
        # initial joint positions (lifting the fingers up)
        self.joint_positions = self.joint_positions

        self.traj_counter = 0  # trajectory waypoint counter

        self.action_counter = 0  # action counter

        # Set action sequence
        if ftpos_deltas is not None:
            self.ftpos_deltas = ftpos_deltas

        # Initial ft pos and vel trajectories
        self.init_x = self.get_ft_pos(self.joint_positions)  # initial fingertip pos
        self.ft_pos_traj = np.expand_dims(self.init_x, 0)
        self.ft_vel_traj = np.zeros((1, 9))

        self.t = 0

        self.done = False

    def set_ft_traj(self, observation):
        # Run this every X timesteps (based on downsampling factor)

        o_pos_cur = observation["object_position"]
        q_cur = observation["robot_position"]
        ft_pos_cur = self.get_ft_pos(q_cur)

        # Get next ft waypoint
        # Get next action
        if self.action_counter < self.ftpos_deltas.shape[0]:
            pred_action = self.ftpos_deltas[self.action_counter, :]
        else:
            pred_action = np.zeros(9)

        # Add ft delta to current ft pos
        ft_pos_next = ft_pos_cur + pred_action

        # Lin interp from current ft pos to next ft waypoint
        # Scale back to meters
        ft_traj = np.stack((ft_pos_cur, ft_pos_next))
        self.ft_pos_traj, self.ft_vel_traj = c_utils.lin_interp_pos_traj(
            ft_traj, self.downsample_time_step, self.time_step
        )

        # Reset traj counter
        self.traj_counter = 0

    def get_ft_des(self, observation):
        """Get fingertip desired pos based on current self.mode"""

        ft_pos_des = self.ft_pos_traj[self.traj_counter, :]
        ft_vel_des = self.ft_vel_traj[self.traj_counter, :]

        if self.traj_counter < len(self.ft_pos_traj) - 1:
            self.traj_counter += 1

        return ft_pos_des, ft_vel_des

    def predict(self, observation):
        """
        Returns torques to command to robot
        """

        # If at end of current trajectory, get new waypoint, or terminate episode
        if self.t == 0 or self.traj_counter >= len(self.ft_pos_traj) - 1:
            self.set_ft_traj(observation)
            self.action_counter += 1

        # Get current waypoints for finger tips
        x_des, dx_des = self.get_ft_des(observation)

        # Get torques from controller
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
