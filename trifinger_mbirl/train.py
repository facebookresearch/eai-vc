# Copyright (c) Facebook, Inc. and its affiliates.
import random
import os
import sys
import torch
import numpy as np
import higher
import matplotlib.pyplot as plt
import argparse
import collections
import wandb

#import trifinger_simulation.finger_types_data

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

import utils.data_utils as d_utils

from trifinger_mbirl.ftpos_mpc import FTPosMPC
from trifinger_mbirl.learnable_costs import *
import trifinger_mbirl.mbirl as mbirl

from trifinger_mbirl.policy import DeterministicPolicy
import trifinger_mbirl.bc as bc

# Set run logging directory to be trifinger_mbirl
mbirl_dir = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(mbirl_dir, "logs/runs")

def main(conf):
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    # Name experiment and make exp directory
    exp_str = get_exp_str(vars(conf))
    exp_dir = os.path.join(conf.log_dir, exp_str)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    # Save conf
    torch.save(conf, f=f'{exp_dir}/conf.pth')

    # Load train and test trajectories
    train_trajs, test_trajs = d_utils.load_trajs(args.file_path, exp_dir)


    if not conf.no_wandb:
        # wandb init
        wandb.init(project='trifinger_mbirl', entity='clairec', name=exp_str, config=conf)


    ### MBIRL training
    if conf.algo == "mbirl":
        time_horizon, n_keypt_dim = train_trajs[0]["ft_pos_cur"].shape # xyz position for each fingertip

        rbf_kernels  = conf.rbf_kernels
        rbf_width    = conf.rbf_width
        cost_type    = conf.cost_type

        # Set learnable cost
        if cost_type == 'Weighted':
            learnable_cost = LearnableWeightedCost(dim=n_keypt_dim)
        elif cost_type == 'TimeDep':
            learnable_cost = LearnableTimeDepWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim)
        elif cost_type == 'RBF':
            learnable_cost = LearnableRBFWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim,
                                                      width=rbf_width, kernels=rbf_kernels)
        elif cost_type == 'Traj':
            learnable_cost = LearnableFullTrajWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim)
        elif cost_type == 'MPTimeDep':
            learnable_cost = LearnableMultiPhaseTimeDepWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim)
        else:
            raise ValueError(f'Cost {cost_type} not implemented')

        # IRL loss
        irl_loss_fn = mbirl.IRLLoss()

        # Run training
        mbirl.train(conf,
                    learnable_cost, 
                    irl_loss_fn,
                    train_trajs, test_trajs,
                    model_data_dir=exp_dir,
                    no_wandb=conf.no_wandb,
                   )
    ### BC training
    elif conf.algo == "bc":
        
        # Make dataset and dataloader
        traindata = bc.ImitationLearningDataset(train_trajs, obs_type=conf.bc_obs_type, device=device)
        dataloader = torch.utils.data.DataLoader(traindata, batch_size=16, shuffle=True)
        
        # Model
        in_dim = traindata[0][0].shape[0]
        out_dim = 9
        policy = DeterministicPolicy(in_dim=in_dim, out_dim=out_dim)

        bc.train(conf, dataloader, policy, exp_dir)
    ### Invalid algo
    else:
        raise ValueError(f"{conf.algo} is invalid -algo")

def get_exp_str(params_dict):
    
    sorted_dict = collections.OrderedDict(sorted(params_dict.items()))

    run_id = params_dict["run_id"]
    file_path = os.path.splitext(os.path.split(params_dict["file_path"])[1])[0]
 
    exp_str = f"exp_{run_id}_{file_path}"

    for key, val in sorted_dict.items():
        # exclude these keys from exp name
        if key in ["file_path", "no_wandb", "log_dir", "run_id", "n_epoch_every_log"]: continue

        # Abbreviate key
        splits = key.split("_")
        short_key = ""
        for split in splits:
            short_key += split[0]
    
        exp_str += "_{}-{}".format(short_key, str(val).replace(".", "p"))

    return exp_str
       

def parse_args():

    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--file_path", default=None, help="""Filepath of trajectory to load""")
    parser.add_argument("--algo", "-a", help="Algorithm to use", choices=["mbirl", "bc"])

    # Optional
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory for run logs")
    parser.add_argument("--no_wandb", action="store_true", help="Don't log in wandb")
    parser.add_argument("--run_id", default="NOID", help="Run ID")
    parser.add_argument("--n_outer_iter", type=int, default=1500, help="Outer loop iterations")
    parser.add_argument("--n_epoch_every_log", type=int, default=100, help="Num epochs every log")


    # MBIRL parameters (used if --algo=="mbirl")
    parser.add_argument("--cost_type", type=str, default="Weighted", help="Learnable cost type",
                        choices=["Weighted", "TimeDep", "RBF", "Traj", "MPTimeDep"])
    parser.add_argument("--cost_lr", type=float, default=0.01, help="Cost learning rate")
    parser.add_argument("--action_lr", type=float, default=0.01, help="Action learning rate")
    parser.add_argument("--n_inner_iter", type=int, default=1, help="Inner loop iterations")
    parser.add_argument("--irl_loss_scale", type=float, default=100, help="IRL loss distance scale")
    # RBF kernel parameters
    parser.add_argument("--rbf_kernels", type=int, default=5, help="Number of RBF kernels")
    parser.add_argument("--rbf_width", type=float, default=2, help="RBF kernel width")


    # BC parameters (used if --algo=="bc")
    parser.add_argument("--bc_lr", type=float, default=1e-3, help="Action learning rate")
    parser.add_argument("--bc_obs_type", type=str, default="goal_rel", help="Observation type",
                        choices=["goal_none", "goal_rel", "img_r3m"])

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

