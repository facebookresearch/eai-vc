import torch
import numpy as np
import sys
import os
import argparse
import random
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import logging

import torchvision.transforms as T
from r3m import load_r3m

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

import utils.data_utils as d_utils

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
    exp_dir, exp_str = get_exp_dir(conf)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    log.info(f"Running experiment with config:\n{OmegaConf.to_yaml(conf)}\n")
    log.info(f"Saving experiment logs in {exp_dir}")

    if not conf.no_wandb:
        # wandb init
        wandb.init(project='trifinger_forward_model', entity='clairec', name=exp_str,
                   config=OmegaConf.to_container(conf, resolve=True, throw_on_missing=True),
                   settings=wandb.Settings(start_method="thread"))

    # Load train and test trajectories for making dataset and for testing MPC
    traj_info = torch.load(conf.demo_path)
    train_trajs = traj_info["train_demos"]
    test_trajs = traj_info["test_demos"]


    r3m = load_r3m("resnet50") # resnet18, resnet34
    r3m.eval()
    r3m.to(device)

    ## DEFINE PREPROCESSING
    transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor()]) # ToTensor() divides by 255



def get_exp_dir(params_dict):
    """
    Get experiment directory to save logs in, and experiment name

    args:
        params_dict: hydra config dict
    return:
        exp_dir: Path of experiment directory
        exp_str: Name of experiment for wandb logging
    """

    hydra_output_dir = HydraConfig.get().runtime.output_dir

    if "experiment" in HydraConfig.get().runtime.choices:
        exp_dir_path = HydraConfig.get().sweep.dir
        exp_id = os.path.basename(os.path.normpath(exp_dir_path))
    else:
        hydra_run_dir = HydraConfig.get().run.dir
        run_date_time = "_".join(hydra_run_dir.split("/")[-2:])
        exp_id = f"single_run_{run_date_time}"

    run_id = params_dict["run_id"]
    demo_path = os.path.splitext(os.path.split(params_dict["demo_path"])[1])[0]
    algo = params_dict["algo"]["name"]

    exp_str = f"{exp_id}_r-{run_id}"

    return hydra_output_dir, exp_str


if __name__ == '__main__':
    main()

