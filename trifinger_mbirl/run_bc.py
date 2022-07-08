# Copyright (c) Facebook, Inc. and its affiliates.
import random
import os
import sys
import torch
import numpy as np
import higher
#import matplotlib.pyplot as plt
import argparse
import collections
import wandb

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from trifinger_mbirl.policy import DeterministicPolicy
import utils.data_utils as d_utils

# Set run logging directory to be trifinger_mbirl
mbirl_dir = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(mbirl_dir, "logs/runs")


class ImitationLearningDataset(torch.utils.data.Dataset):
    def __init__(self, demos):
        self.dataset = []
        for demo in demos:
            num_obs = demo['o_cur_pos'].shape[0]
            for i in range(num_obs):
                obs = torch.cat([torch.FloatTensor(demo['o_cur_pos'][i]), torch.FloatTensor(demo['ft_pos_cur'][i])])
                action = torch.FloatTensor(demo['delta_ftpos'][i])
                self.dataset.append((obs, action))
        # self.dataset = [
        #     (torch.FloatTensor(x[i]), torch.FloatTensor(y[i])) for i in range(len(x))
        # ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def plot_loss(loss_dict, outer_i):
    log_dict = {f'{k}': v for k, v in loss_dict.items()}
    log_dict['outer_i'] = outer_i
    wandb.log(log_dict)


def main(conf):
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)

    # Load trajectory, get x_init and time_horizon
    data = np.load(conf.file_path, allow_pickle=True)["data"]
    traj = d_utils.get_traj_dict_from_obs_list(data)
    
    # Full trajectory, downsampled by 75x (3.3Hz)
    traj = d_utils.downsample_traj_dict(traj, new_time_step=0.3)
        
    time_horizon = traj["ft_pos_cur"].shape[0]
    conf.time_horizon = time_horizon

    exp_str = get_exp_str(vars(conf))
    exp_dir = os.path.join(conf.log_dir, exp_str)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if not conf.no_wandb:
        # wandb init
        wandb.init(project='trifinger_mbirl', entity='fmeier', name=exp_str, config=conf)


    bc_loss = torch.nn.MSELoss()

    # For now, just use one traj for training and testing
    train_trajs = [traj]
    test_trajs = [traj]
    traindata = ImitationLearningDataset(train_trajs)
    dataloader = torch.utils.data.DataLoader(traindata, batch_size=16, shuffle=True)
    policy = DeterministicPolicy(in_dim=12, out_dim=9)

    print("pause")
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for i in range(100):
        for batch, (obs, actions) in enumerate(dataloader):
            optimizer.zero_grad()
            pred_actions = policy(obs)
            loss = bc_loss(pred_actions, actions)
            loss.backward()
            optimizer.step()
        print(f"i: {i}, loss: {loss.item()}")


    torch.save({
        'bc_loss_train' : loss,
        'policy': policy.state_dict(),
        'conf'           : conf,
    }, f=f'{exp_dir}/bc_log.pth')

def get_exp_str(params_dict):
    
    sorted_dict = collections.OrderedDict(sorted(params_dict.items()))

    run_id = params_dict["run_id"]
    exp_str = f"exp_{run_id}"
    for key, val in sorted_dict.items():
        # exclude these keys from exp name
        if key in ["file_path", "no_wandb", "log_dir", "run_id"]: continue

        # Abbreviate key
        splits = key.split("_")
        short_key = ""
        for split in splits:
            short_key += split[0]
    
        exp_str += "_{}-{}".format(short_key, str(val).replace(".", "p"))

    return exp_str
       

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", default=None, help="""Filepath of trajectory to load""")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory for run logs")
    parser.add_argument("--no_wandb", action="store_true", help="Don't log in wandb")
    parser.add_argument("--run_id", default="BC", help="Run ID")

    # Parameters
    parser.add_argument("--cost_type", type=str, default="Weighted", help="Learnable cost type")
    parser.add_argument("--cost_lr", type=float, default=0.01, help="Cost learning rate")
    parser.add_argument("--action_lr", type=float, default=0.01, help="Action learning rate")
    parser.add_argument("--n_outer_iter", type=int, default=1500, help="Outer loop iterations")
    parser.add_argument("--n_inner_iter", type=int, default=1, help="Inner loop iterations")
    parser.add_argument("--irl_loss_scale", type=float, default=100, help="IRL loss distance scale")

    # RBF kernel parameters
    parser.add_argument("--rbf_kernels", type=int, default=5, help="Number of RBF kernels")
    parser.add_argument("--rbf_width", type=float, default=2, help="RBF kernel width")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

