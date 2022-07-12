import os
import sys
import numpy as np
import enum
import torch

import trifinger_simulation.finger_types_data
import trifinger_simulation.pinocchio_utils

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from control.impedance_controller import ImpedanceController
from control.custom_pinocchio_utils import CustomPinocchioUtils
import control.cube_utils as c_utils
from trifinger_mbirl.policy import DeterministicPolicy

class BCPolicy:
    """
    Follow fingertip position trajectory
    """

    def __init__(self, ckpt_path, expert_actions, action_space, platform, time_step=0.001,
                 downsample_time_step=0.001, total_ep_steps=1000):
        """
        ftpos_traj: fingertip position trajectory [T, 9]
        """

        self.action_space = action_space
        self.time_step = time_step
        self.total_ep_steps = total_ep_steps
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

        self.traj_counter = 0

        self.policy = self.load_policy(ckpt_path)

        self.expert_actions = expert_actions

    def reset(self):
        # initial joint positions (lifting the fingers up)
        self.joint_positions = self.joint_positions

        # mode and trajectory initializations
        self.traj_counter = 0

        self.set_traj_counter = 0

        # Initial ft pos and vel trajectories
        self.init_x = self.get_ft_pos(self.joint_positions) # initial fingertip pos
        self.ft_pos_traj = np.expand_dims(self.init_x, 0)
        self.ft_vel_traj = np.zeros((1,9))

        self.t = 0

        self.done = False

    def load_policy(self, ckpt_path):
        policy = DeterministicPolicy(in_dim=12, out_dim=9)
        info = torch.load(ckpt_path)
        policy.load_state_dict(info["policy"])
        return policy

    def set_ft_traj(self, observation):
        # Run this every X timesteps (based on downsampling factor)

        o_pos_cur = observation["object_observation"]["position"]
        q_cur = observation["robot_observation"]["position"]
        ft_pos_cur = self.get_ft_pos(q_cur)

        # Get next ft waypoint
        # Get ft delta from policy
        obs = torch.cat([torch.FloatTensor(o_pos_cur), torch.FloatTensor(ft_pos_cur)])
        pred_action = self.policy(obs).detach().numpy()
    
        # Use expert actions
        #pred_action = self.expert_actions[self.set_traj_counter, :]

        # Add ft delta to current ft pos
        ft_pos_next = ft_pos_cur + pred_action

        # Lin interp from current ft pos to next ft waypoint
        ft_traj = np.stack((ft_pos_cur, ft_pos_next))
        self.ft_pos_traj, self.ft_vel_traj = self.interp_ft_traj(ft_traj)

        # Reset traj counter
        self.traj_counter = 0

        self.set_traj_counter += 1

    def interp_ft_traj(self, ft_pos_traj_in):
        """
        Interpolate between waypoints in ftpos trajectory, and compute velocities
        For now, just try linear interpolation between waypoints, with first-order hold on linear velocity between waypoints
        """
        ft_pos_traj =  c_utils.lin_interp_waypoints(ft_pos_traj_in, self.downsample_time_step,
                                                    time_step_out=self.time_step)
        
        ft_vel_traj = np.zeros(ft_pos_traj.shape)
        for i in range(ft_pos_traj.shape[0] - 1):
            v = (ft_pos_traj[i+1,:] - ft_pos_traj[i,:]) / self.time_step
            ft_vel_traj[i, :] = v

        return ft_pos_traj, ft_vel_traj

    def get_ft_des(self, observation):
        """ Get fingertip desired pos  """

        ft_pos_des = self.ft_pos_traj[self.traj_counter, :]
        ft_vel_des = self.ft_vel_traj[self.traj_counter, :]

        if self.traj_counter < len(self.ft_pos_traj) - 1:
            self.traj_counter += 1

        return ft_pos_des, ft_vel_des

    def predict(self, observation):
        """
        Returns torques to command to robot
        """
    
        if self.t >= self.total_ep_steps:
            self.done = True

        if self.t == 0 or self.traj_counter >= len(self.ft_pos_traj) -1:
            self.set_ft_traj(observation)
            #print(self.set_traj_counter)

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

