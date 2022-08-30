import torch
import numpy as np
import argparse
import os
import sys

from r3m import load_r3m

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from trifinger_mbirl.forward_models.train_phase2_model import Phase2Model
from trifinger_mbirl.forward_models.models.phase1_model import Phase1Model
import utils.data_utils as d_utils


# Compute next state given current state and action (ft position deltas)
class TwoPhaseMPC(torch.nn.Module):

    def __init__(self, time_horizon, phase2_model_path, f_num=3, mode=None):
        super().__init__()
        self.time_horizon = time_horizon
        self.f_num = f_num
        self.n_keypt_dim = self.f_num * 3
        self.a_dim = self.f_num * 3
        self.action_seq = torch.nn.Parameter(torch.Tensor(np.zeros([time_horizon, self.a_dim])))

        self.mode = mode

        self.phase1_model = Phase1Model()

        # Load phase 2 model weights
        phase2_model_dict = torch.load(phase2_model_path) 
        self.in_dim = phase2_model_dict["in_dim"]
        self.out_dim = phase2_model_dict["out_dim"]
        hidden_dims = phase2_model_dict["hidden_dims"]
        self.phase2_model = Phase2Model(self.in_dim, self.out_dim, hidden_dims)
        self.phase2_model.load_state_dict(phase2_model_dict["model_state_dict"])
        self.phase2_start_ind = phase2_model_dict["phase2_start_ind"]

        # Freeze network params
        for name, param in self.phase2_model.named_parameters():
            param.requires_grad = False

        self.obj_state_type = phase2_model_dict["conf"].obj_state_type

    def forward(self, obs_dict, action=None):
        """ 
        Given current state and action, and mode, compute next state

        args:
            obs_dict = {
                        "ft_state": ft positions,
                        "o_state": object state,
                        "mode":  mode,
                       }
            action
    
        return:
        """

        if action is None:
            return torch.cat([obs_dict["ft_state"], obs_dict["o_state"]], dim=1)
            #obs_dict["action"] = torch.FloatTensor(np.zeros((1, self.a_dim)))
        else:
            #if obs_dict["mode"] == 2: action = action * 10 # TODO for testing model with scaled actions
            obs_dict["action"] = torch.unsqueeze(action, 0)

        if obs_dict["mode"] == 1:
            x_next = self.phase1_model(obs_dict)
        else: 
            # Mode 2
            x_next = self.phase2_model(obs_dict)
        
        return x_next

    def roll_out(self, obs_dict_init):
        """ Given intial state, compute trajectory of length self.time_horizon with actions self.action_seq """
        pred_traj = []
        x_next = self.forward(obs_dict_init)

        ft_state = x_next[:, :self.a_dim]
        o_state = x_next[:, self.a_dim:]
        t_ind = 0

        if self.mode is None:
            if t_ind < self.phase2_start_ind: mode = 1
            else: mode = 2
        else:
            mode = self.mode

        obs_dict_next = {
                         "ft_state": ft_state,
                         "o_state" : o_state,
                         "mode"    : mode,
                        }
        pred_traj.append(torch.squeeze(x_next.clone()))

        for t in range(self.time_horizon):
            a = self.action_seq[t]
            x_next = self.forward(obs_dict_next, a)
            x_next = self.clip_ftpos(x_next) # TODO implement clipping

            pred_traj.append(torch.squeeze(x_next.clone()))

            ft_state = x_next[:, :self.a_dim]
            o_state = x_next[:, self.a_dim:]

            t_ind += 1
            if self.mode is None:
                if t_ind < self.phase2_start_ind: mode = 1
                else: mode = 2
            else:
                mode = self.mode

            obs_dict_next = {
                             "ft_state": ft_state,
                             "o_state" : o_state,
                             "mode"    : mode,
                            }


        pred_traj = torch.stack(pred_traj)
        return pred_traj

    def roll_out_gt_state(self, expert_traj):
        """ [FOR TESTING] Apply action to ground truth state at each timestep (don't use predicted next state as new initial state) """
        pred_traj = []

        obs_dict = d_utils.get_obs_dict_from_traj(expert_traj, 0, self.obj_state_type)
        x_next = self.forward(obs_dict)
        pred_traj.append(torch.squeeze(x_next.clone()))

        for t in range(self.time_horizon):
            obs_dict = d_utils.get_obs_dict_from_traj(expert_traj, t, self.obj_state_type)
            a = self.action_seq[t]
            x_next = self.forward(obs_dict, a)

            pred_traj.append(torch.squeeze(x_next.clone()))

        pred_traj = torch.stack(pred_traj)
        return pred_traj

    def reset_actions(self):
        self.action_seq.data = torch.Tensor(np.zeros([self.time_horizon, self.a_dim]))

    def clip_ftpos(self, ftpos):
        ftpos_min = [-20] * self.out_dim
        ftpos_max = [20] * self.out_dim

        ftpos = torch.Tensor(ftpos)
        ftpos_min = torch.unsqueeze(torch.Tensor(ftpos_min), 0)
        ftpos_max = torch.unsqueeze(torch.Tensor(ftpos_max), 0)

        ftpos = torch.where(ftpos > ftpos_max, ftpos_max, ftpos)
        ftpos = torch.where(ftpos < ftpos_min, ftpos_min, ftpos)

        return ftpos

    def set_action_seq_for_testing(self, action_seq):
        self.action_seq.data = torch.Tensor(action_seq)

def main(args):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load r3m model
    r3m = load_r3m("resnet50")  # resnet18, resnet34
    r3m.eval()
    r3m.to(device)

    # Load demo.npz
    data = np.load(args.demo_path, allow_pickle=True)["data"]
    traj = d_utils.get_traj_dict_from_obs_list(data, scale=100)
    traj = d_utils.downsample_traj_dict(traj, new_time_step=0.2)

    time_horizon = traj["ft_pos_cur"].shape[0]

    mpc = TwoPhaseMPC(time_horizon-1, args.phase2_model_path)
    phase2_start_ind = mpc.phase2_start_ind

    obs_dict_init = d_utils.get_obs_dict_from_traj(traj, 0, mpc.obj_state_type)

    # Run roll_out to get trajectory from initial state
    mpc.set_action_seq_for_testing(traj["delta_ftpos"])
    pred_traj = mpc.roll_out(obs_dict_init)
    #pred_traj = mpc.roll_out_gt_state(traj) # one-step rollout (from ground truth states)

    pred_traj = pred_traj.detach().numpy()
    pred_ft_pos = pred_traj[:, :mpc.a_dim]
    pred_o_state = pred_traj[:, mpc.a_dim:]
        
    if mpc.obj_state_type == "pos":
        true_o_state = traj["o_pos_cur"]
    elif mpc.obj_state_type == "vertices":
        true_o_state = traj["vertices"]
    elif mpc.obj_state_type == "img_r3m":
        true_o_state = traj["image_60_r3m"]
    else:
        raise ValueError()

    # Compute l2 distance between each ft and object state (pos or vertices)
    ft_pos_err = np.zeros((time_horizon, 3))
    for i in range(3):
        per_finger_err = np.linalg.norm((pred_ft_pos[:, i*3:i*3+3] - traj["ft_pos_cur"][:, i*3:i*3+3]), axis=1)
        ft_pos_err[:, i] = per_finger_err

    if mpc.obj_state_type == "pos":
        o_state_err = np.expand_dims(np.linalg.norm((pred_o_state - true_o_state), axis=1), 1)
    elif mpc.obj_state_type == "vertices":
        o_state_err = np.zeros((time_horizon, 8))
        for i in range(8):
            per_vert_err = np.linalg.norm((pred_o_state[:, i*3:i*3+3] - true_o_state[:, i*3:i*3+3]), axis=1)
            o_state_err[:, i] = per_vert_err
    elif mpc.obj_state_type == "img_r3m":
        o_state_err = np.expand_dims(np.linalg.norm((pred_o_state - true_o_state), axis=1), 1)
    else:
        raise ValueError()

    # Compare against ground truth trajectory to check that rollout is correct
    #d_utils.plot_traj(
    #        "ft position (cm)", 
    #        None,
    #        ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
    #        {
    #        "pred":  {"y": pred_ft_pos, "x": traj["t"], "marker": "x"},
    #        "demo": {"y": traj["ft_pos_cur"], "x": traj["t"]},
    #        },
    #        plot_timestamp = traj["t"][phase2_start_ind]
    #        )

    ## Object state
    #if mpc.obj_state_type == "pos":
    #    d_utils.plot_traj(
    #            "object position (cm)", 
    #            None,
    #            ["x", "y", "z"],
    #            {
    #            "pred":  {"y": pred_o_state, "x": traj["t"], "marker": "x"},
    #            "demo": {"y":  true_o_state, "x": traj["t"]},
    #            },
    #            plot_timestamp = traj["t"][phase2_start_ind]
    #            )
    #elif mpc.obj_state_type == "vertices":
    #    for i in range(8):
    #        d_utils.plot_traj(
    #                f"object vertex {i}", 
    #                None,
    #                ["x", "y", "z"],
    #                {
    #                "pred":  {"y": pred_o_state[:, i*3:i*3+3], "x": traj["t"], "marker": "x"},
    #                "demo": {"y":  true_o_state[:, i*3:i*3+3], "x": traj["t"]},
    #                },
    #                plot_timestamp = traj["t"][phase2_start_ind]
    #                )
    #else:
    #    raise ValueError()
       
    print(np.amax(ft_pos_err, axis=0))
    print(np.amax(o_state_err, axis=0))

    # Plot l2 distance between each ft and object state (pos or vertices)
    d_utils.plot_traj(
            "ft position error (cm)", 
            None,
            ["f1", "f2", "f3"],
            {
            "err":  {"y": ft_pos_err, "x": traj["t"], "marker": "x"},
            },
            plot_timestamp = traj["t"][phase2_start_ind]
            )

    # Object state
    if mpc.obj_state_type == "pos":
        d_utils.plot_traj(
                "object position error (cm)", 
                None,
                ["pos"],
                {
                "err":  {"y": o_state_err, "x": traj["t"], "marker": "x"},
                },
                plot_timestamp = traj["t"][phase2_start_ind]
                )
    elif mpc.obj_state_type == "vertices":
        d_utils.plot_traj(
                f"object vertex position error (cm)", 
                None,
                ["v0","v1", "v2", "v3", "v4", "v5", "v6", "v7"],
                {
                "err":  {"y": o_state_err, "x": traj["t"], "marker": "x"},
                },
                plot_timestamp = traj["t"][phase2_start_ind]
                )
    elif mpc.obj_state_type == "img_r3m":
        d_utils.plot_traj(
                "R3M embedding L2 distance", 
                None,
                ["r3m"],
                {
                "err":  {"y": o_state_err, "x": traj["t"], "marker": "x"},
                },
                plot_timestamp = traj["t"][phase2_start_ind]
                )
    else:
        raise ValueError()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2_model_path", "-m", default=None, help="""Filepath of phase2 model to load""")
    parser.add_argument("--demo_path", "-d", default=None, help="""Filepath of demo to load for test""")
    args = parser.parse_args()

    main(args)

