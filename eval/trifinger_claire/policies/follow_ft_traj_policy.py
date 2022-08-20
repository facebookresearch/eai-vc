import os
import sys
import numpy as np
import enum

import trifinger_simulation.finger_types_data
import trifinger_simulation.pinocchio_utils

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from control.impedance_controller import ImpedanceController
from control.custom_pinocchio_utils import CustomPinocchioUtils
import control.cube_utils as c_utils

class FollowFtTrajPolicy:
    """
    Follow fingertip position trajectory
    """

    def __init__(self, ft_pos_traj_in, action_space, platform, time_step=0.001,
                 downsample_time_step=0.001):
        """
        ftpos_traj: fingertip position trajectory [T, 9]
        """

        self.action_space = action_space
        self.time_step = time_step
        self.ft_pos_traj_in = ft_pos_traj_in
        self.downsample_time_step = downsample_time_step

        # TODO hardcoded
        robot_properties_path = "../trifinger_simulation/trifinger_simulation/robot_properties_fingers"

        urdf_file = trifinger_simulation.finger_types_data.get_finger_urdf("trifingerpro")

        finger_urdf_path = os.path.join(robot_properties_path, "urdf", urdf_file)

        # initial joint positions (lifting the fingers up)
        self.joint_positions = np.array([-0.08, 1.15, -1.5] * 3) # "down and out" position

        # set platform (robot)
        self.platform = platform

        self.Nf = 3 # Number of fingers
        self.Nq = self.Nf * 3 # Number of joints in hand

        # class with kinematics functions
        self.kinematics = CustomPinocchioUtils(
                self.platform.simfinger.finger_urdf_path,
                self.platform.simfinger.tip_link_names,
                self.platform.simfinger.link_names)

        self.controller = ImpedanceController(self.kinematics)

    def reset(self, ft_pos_traj_in=None):
        # initial joint positions (lifting the fingers up)
        self.joint_positions = self.joint_positions

        # mode and trajectory initializations
        self.traj_counter = 0

        # Initial ft pos and vel trajectories
        if ft_pos_traj_in is not None:
            self.ft_pos_traj_in = ft_pos_traj_in
        self.ft_pos_traj, self.ft_vel_traj = c_utils.lin_interp_pos_traj(self.ft_pos_traj_in,
                                                                         self.downsample_time_step,
                                                                         self.time_step)

        self.t = 0

        self.done = False

    def get_ft_des(self, observation):
        """ Get fingertip desired pos based on current self.mode """

        ft_pos_des = self.ft_pos_traj[self.traj_counter, :]
        ft_vel_des = self.ft_vel_traj[self.traj_counter, :]

        if self.traj_counter < len(self.ft_pos_traj) - 1:
            self.traj_counter += 1
        else:   
            self.done = True

        return ft_pos_des, ft_vel_des

    def predict(self, observation):
        """
        Returns torques to command to robot
        """

        # 3. Get current waypoints for finger tips
        x_des, dx_des = self.get_ft_des(observation)

        #4. Get torques from controller
        q_cur = observation["robot_observation"]["position"]
        dq_cur = observation["robot_observation"]["velocity"]
        torque = self.controller.get_command_torque(x_des, dx_des, q_cur, dq_cur)
    
        self.t += 1

        return self.clip_to_space(torque)

    def clip_to_space(self, action):
        """ Clip action to action space """

        return np.clip(action, self.action_space.low, self.action_space.high)

    def get_ft_pos(self, q):
        """ Get fingertip positions given current joint configuration q """

        ft_pos = np.array(self.kinematics.forward_kinematics(q)).reshape(self.Nq)
        return ft_pos

    def get_observation(self):

        obs = {"policy": 
                {
                "controller": self.controller.get_observation(),
                "t" : self.t,
                }
              }

        return obs

