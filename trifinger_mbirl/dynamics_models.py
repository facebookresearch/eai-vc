import torch
import os
import sys

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from envs.cube_env import SimCubeEnv, ActionType

SIM_TIME_STEP = 0.004

class DiffTrifingerSim(torch.nn.Module):

    def __init__(self):
        super(DiffTrifingerSim, self).__init__()
        env = SimCubeEnv(
            goal_pose=None,  # passing None to sample a random trajectory
            action_type=ActionType.TORQUE,
            visualization=False,
            no_collisions=args.no_collisions,
            enable_cameras=True,
            finger_type="trifingerpro",
            time_step=SIM_TIME_STEP,
        )


    def forward(self, action):
        return