import torch
import numpy as np

import argparse
import os
import sys

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

import utils.data_utils as d_utils

# Compute next state given current state and action (ft position deltas)
class FTPosMPC(torch.nn.Module):

    def __init__(self, time_horizon, f_num=3):
        super().__init__()
        self.time_horizon = time_horizon
        self.f_num = f_num
        self.n_keypt_dim = self.f_num * 3
        self.a_dim = self.f_num * 3
        self.action_seq = torch.nn.Parameter(torch.Tensor(np.zeros([time_horizon, self.a_dim])))

    def forward(self, x, u=0):
        """ 
        Given current state and action, compute next state

        args:
            x: current state (ft_pos)
            u: action (delta ftpos)
    
        return:
            x_next: next state (ft_pos)
        """

        x_next = x + u
        # Clip ft positions
        x_next = self.clip_ftpos(x_next)

        return x_next

    def roll_out(self, x_init):
        """ Given intial state, compute trajectory of length self.time_horizon with actions self.action_seq """
        x_traj = []
        x_next = self.forward(x_init)
        x_traj.append(x_next)

        for t in range(self.time_horizon):
            a = self.action_seq[t]
            x_next = self.forward(x_next, a)
            x_next = self.clip_ftpos(x_next)
            x_traj.append(x_next.clone())

        return torch.stack(x_traj)

    def reset_actions(self):
        self.action_seq.data = torch.Tensor(np.zeros([self.time_horizon, self.a_dim]))

    def clip_ftpos(self, ftpos):
        arena_r = 0.2 # arena radius
        ftpos_min = [-arena_r, -arena_r, 0.] * 3
        ftpos_max = [arena_r, arena_r, 0.12] * 3

        ftpos = torch.Tensor(ftpos)
        ftpos_min = torch.Tensor(ftpos_min)
        ftpos_max = torch.Tensor(ftpos_max)

        ftpos_clipped = torch.where(ftpos > ftpos_max, ftpos_max, ftpos)
        ftpos_clipped = torch.where(ftpos < ftpos_min, ftpos_min, ftpos)

        return ftpos_clipped

    def set_action_seq_for_testing(self, action_seq):
        self.action_seq.data = torch.Tensor(action_seq)


def main(args):
    # Load demo.npz
    data = np.load(args.file_path, allow_pickle=True)["data"]
    traj = d_utils.get_traj_dict_from_obs_list(data)
    #traj = d_utils.crop_traj_dict(d_utils.downsample_traj_dict(traj), [0, 30])
    traj = d_utils.downsample_traj_dict(traj)
    
    ft_pos_cur = traj["ft_pos_cur"]
    delta_ftpos = traj["delta_ftpos"]
    
    time_horizon = ft_pos_cur.shape[0]
    x_init = ft_pos_cur[0, :]
    
    # Run roll_out to get trajectory from initial state
    ftpos_mpc = FTPosMPC(time_horizon-1)
    ftpos_mpc.set_action_seq_for_testing(delta_ftpos)
    x_traj = ftpos_mpc.roll_out(x_init)
    
    x_traj = x_traj.detach().numpy()
    
    # Compare against ground truth trajectory to check that rollout is correct
    d_utils.plot_traj(
            "ft position", 
            None,
            ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
            {
            "pred":  {"y": x_traj, "x": traj["t"], "marker": "x"},
            "demo": {"y": ft_pos_cur, "x": traj["t"]},
            })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", default=None, help="""Filepath of trajectory to load""")
    args = parser.parse_args()
    main(args)

