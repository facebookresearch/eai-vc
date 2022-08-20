#w Copyright (c) Facebook, Inc. and its affiliates.
import random
import os
import sys
import torch
import numpy as np
import higher
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import logging

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

import utils.data_utils as d_utils
import utils.train_utils as t_utils
from trifinger_mbirl.mbirl import MBIRL
import trifinger_mbirl.bc as bc
import trifinger_mbirl.bc_finetune as bc_finetune
from trifinger_mbirl.policy import DeterministicPolicy

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(conf):

    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Name experiment and make exp directory
    exp_dir, exp_str, exp_id = t_utils.get_exp_dir(conf)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    log.info(f"Running experiment with config:\n{OmegaConf.to_yaml(conf)}\n")
    log.info(f"Saving experiment logs in {exp_dir}")
    
    # Save conf
    torch.save(conf, f=f'{exp_dir}/conf.pth')

    # Load train and test trajectories
    traj_info = torch.load(conf.demo_path)
    torch.save(traj_info, f=f"{exp_dir}/demo_info.pth") # Re-save trajectory info in exp_dir
    train_trajs = traj_info["train_demos"]
    test_trajs = traj_info["test_demos"]

    if not conf.no_wandb:
        conf_for_wandb = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)
        conf_for_wandb["exp_id"] = exp_id # Add experiment id to conf, for wandb
        # wandb init
        wandb.init(project='trifinger_mbirl', entity='clairec', name=exp_str,
                   config = conf_for_wandb,
                   settings=wandb.Settings(start_method="thread"))
    
    ### MBIRL training
    if conf.algo.name == "mbirl":

        mbirl = MBIRL(conf.algo, traj_info)
        mbirl.train(model_data_dir=exp_dir, no_wandb=conf.no_wandb)

    ### BC training
    elif conf.algo.name == "bc":
        
        # Make dataset and dataloader
        traindata = bc.ImitationLearningDataset(train_trajs, obs_type=conf.algo.obs_type, device=device)
        dataloader = torch.utils.data.DataLoader(traindata, batch_size=16, shuffle=True)
        
        # Model
        in_dim = traindata[0][0].shape[0]
        out_dim = 9
        policy = DeterministicPolicy(in_dim=in_dim, out_dim=out_dim, device=device)

        bc.train(conf.algo, dataloader, policy, exp_dir)

    ### BC with finetuning training
    elif conf.algo.name == "bc_finetune":

        # Make dataset and dataloader
        traindata = bc_finetune.ImitationLearningDataset(train_trajs, device=device)
        dataloader = torch.utils.data.DataLoader(traindata, batch_size=16, shuffle=True)

        # Model
        in_dim = 4105 #traindata[0][0].shape[0]
        out_dim = 9
        policy = DeterministicPolicy(in_dim=in_dim, out_dim=out_dim, device=device)

        bc_finetune.train(conf.algo, dataloader, policy, exp_dir)

    ### Invalid algo
    else:
        raise ValueError(f"{conf.algo.name} is invalid algo")

if __name__ == '__main__':
    main()

