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
    def __init__(self, demos, obs_type="goal_none"):
        self.dataset = []
        for demo in demos:
            num_obs = demo['o_pos_cur'].shape[0]

            for i in range(num_obs):
                obs_dict = {
                            "o_pos_cur" : demo["o_pos_cur"][i],
                            "ft_pos_cur": demo["ft_pos_cur"][i],
                            "o_pos_des" : demo["o_pos_des"][0, :] # Goal object position
                           }

                obs = get_bc_obs(obs_dict, obs_type)

                action = torch.FloatTensor(demo['delta_ftpos'][i])

                self.dataset.append((obs, action))

        # TODO make obs relative to goal (final, and intermmediate)

        # self.dataset = [
        #     (torch.FloatTensor(x[i]), torch.FloatTensor(y[i])) for i in range(len(x))
        # ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_bc_obs(obs_dict, obs_type):
    """
    obs_type (str): [
                     "goal_none", # fingertip pos and object pos in world frame, no goal [12]
                     "goal_rel", # fingertip pos and object pos, relative to object goal [12]
                    ]
    """

    if obs_type == "goal_none":
        obs = torch.cat([torch.FloatTensor(obs_dict["o_pos_cur"]), torch.FloatTensor(obs_dict["ft_pos_cur"])])

    elif obs_type == "goal_rel":
        ft_pos_rel = np.repeat(obs_dict["o_pos_des"], 3) - obs_dict["ft_pos_cur"]
        o_pos_rel  = obs_dict["o_pos_des"] - obs_dict["o_pos_cur"]

        obs = torch.cat([torch.FloatTensor(ft_pos_rel), torch.FloatTensor(o_pos_rel)])

    else:
        raise ValueError("Invalid obs_type")

    return obs

def plot_loss(loss_dict, outer_i):
    log_dict = {f'{k}': v for k, v in loss_dict.items()}
    log_dict['outer_i'] = outer_i
    wandb.log(log_dict)

def train(conf, dataloader, policy, model_data_dir):

    # Make logging directories
    ckpts_dir = os.path.join(model_data_dir, "ckpts") 
    if not os.path.exists(ckpts_dir): os.makedirs(ckpts_dir)

    bc_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(policy.parameters(), lr=conf.lr)

    for outer_i in range(conf.n_outer_iter):
        for batch, (obs, actions) in enumerate(dataloader):
            optimizer.zero_grad()
            pred_actions = policy(obs)
            loss = bc_loss(pred_actions, actions)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {outer_i}, loss: {loss.item()}")

        if (outer_i+1) % 25 == 0:

            torch.save({
                'bc_loss_train' : loss,
                'policy'        : policy.state_dict(),
                'conf'          : conf,
            }, f=f'{ckpts_dir}/epoch_{outer_i+1}_ckpt.pth')


def main(conf):
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)
        
    # Name experiment and make exp directory
    exp_str = get_exp_str(vars(conf))
    exp_dir = os.path.join(conf.log_dir, exp_str)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Save conf
    conf.algo = "bc"
    torch.save(conf, f=f'{exp_dir}/conf.pth')

    # Load train and test trajectories
    train_trajs, test_trajs = d_utils.load_trajs(args.file_path, exp_dir)

    if not conf.no_wandb:
        # wandb init
        wandb.init(project='trifinger_mbirl', entity='fmeier', name=exp_str, config=conf)


    # Make dataset and dataloader
    traindata = ImitationLearningDataset(train_trajs, obs_type=conf.bc_obs_type)
    dataloader = torch.utils.data.DataLoader(traindata, batch_size=16, shuffle=True)
    
    # Model
    in_dim = traindata[0][0].shape[0]
    out_dim = 9
    policy = DeterministicPolicy(in_dim=in_dim, out_dim=out_dim)

    train(conf, dataloader, policy, exp_dir)

def get_exp_str(params_dict):
    
    sorted_dict = collections.OrderedDict(sorted(params_dict.items()))

    run_id = params_dict["run_id"]
    file_path = os.path.splitext(os.path.split(params_dict["file_path"])[1])[0]
 
    exp_str = f"exp_{run_id}_{file_path}"

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
    parser.add_argument("--lr", type=float, default=1e-3, help="Action learning rate")
    parser.add_argument("--n_outer_iter", type=int, default=200, help="Outer loop iterations")
    parser.add_argument("--bc_obs_type", type=str, default="goal_none", help="Observation type",
                        choices=["goal_none", "goal_rel"])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

