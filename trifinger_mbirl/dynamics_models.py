import torch
import os
import sys
import numpy as np

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from envs.cube_env import SimCubeEnv, ActionType
import control.cube_utils as c_utils
from control.impedance_controller import ImpedanceController
from control.custom_pinocchio_utils import CustomPinocchioUtils

SIM_TIME_STEP = 0.004


class DiffTrifingerSim(torch.nn.Module):

    def __init__(self, start_pose, goal_pose, downsample_time):
        super(DiffTrifingerSim, self).__init__()
        self.env = SimCubeEnv(
            goal_pose=None,  # passing None to sample a random trajectory
            action_type=ActionType.TORQUE,
            visualization=False,
            no_collisions=True,
            enable_cameras=True,
            finger_type="trifingerpro",
            time_step=SIM_TIME_STEP,
        )

        self.env.reset(goal_pose_dict=goal_pose, init_pose_dict=start_pose)

        self.downsample_time_step = downsample_time
        self.time_step = 0
        self.cur_state = torch.zeros(2)

        # class with kinematics functions
        self.kinematics = CustomPinocchioUtils(
            self.env.platform.simfinger.finger_urdf_path,
            self.env.platform.simfinger.tip_link_names,
            self.env.platform.simfinger.link_names)

        self.controller = ImpedanceController(self.kinematics)

    # Franzi: copied from bc_policy
    def interp_ft_traj(self, ft_pos_traj_in):
        """
        Interpolate between waypoints in ftpos trajectory, and compute velocities
        For now, just try linear interpolation between waypoints,
        with zero-order hold on linear velocity between waypoints
        """
        ft_pos_traj = c_utils.lin_interp_waypoints(ft_pos_traj_in, self.downsample_time_step,
                                                   time_step_out=self.time_step)

        ft_vel_traj = np.zeros(ft_pos_traj.shape)
        for i in range(ft_pos_traj.shape[0] - 1):
            v = (ft_pos_traj[i + 1, :] - ft_pos_traj[i, :]) / self.time_step
            ft_vel_traj[i, :] = v

        return ft_pos_traj, ft_vel_traj

    # actions are finger tip deltas
    def forward(self, action):

        ft_pos_next_des = self.ft_pos_cur + action

        # Lin interp from current ft pos to next ft waypoint
        ft_start_goal = np.stack((self.ft_pos_cur, ft_pos_next_des))
        ft_pos_traj, ft_vel_traj = self.interp_ft_traj(ft_start_goal)

        for i in range(len(ft_pos_traj)):

            # 3. Get current waypoints for finger tips
            x_des, dx_des = ft_pos_traj[i, :], ft_vel_traj[i, :] #self.get_ft_des(observation)

            #4. Get torques from controller
            q_cur = observation["robot_observation"]["position"]
            dq_cur = observation["robot_observation"]["velocity"]
            torque = self.controller.get_command_torque(x_des, dx_des, q_cur, dq_cur)


        return