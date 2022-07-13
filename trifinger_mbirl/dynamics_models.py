import torch
import os
import sys

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from envs.cube_env import SimCubeEnv, ActionType
import control.cube_utils as c_utils

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

        self.downsample_time = downsample_time
        self.time_step = 0
        self.cur_state = torch.zeros(2)



    def forward(self, action):
        upsampled_actions = c_utils.lin_interp_waypoints(ft_pos_traj_in, self.downsample_time_step,
                                                           time_step_out=self.time_step)

        return