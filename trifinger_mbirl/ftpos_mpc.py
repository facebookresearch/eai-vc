import torch
import numpy as np

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

        for t in range(self.time_horizon-1):
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

# TODO write script to test this with demo traj

