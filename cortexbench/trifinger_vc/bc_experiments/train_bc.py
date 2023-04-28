# w Copyright (c) Meta Platforms, Inc. and affiliates.
import random
import os
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
import logging
import json

import trifinger_vc.utils.train_utils as t_utils
from trifinger_vc.algos.bc_finetune import BCFinetune

# A logger for this file
log = logging.getLogger(__name__)

"""
Main script to launch imitation learning training
(mbirl, policy_opt, bc, bc_finetune)
"""


@hydra.main(version_base=None, config_path="../src/trifinger_vc/config", config_name="bc_default")
def main(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Name experiment and make exp directory
    exp_dir, exp_str, exp_id = t_utils.get_exp_dir(conf)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    exp_id = f"{conf.task.name}_{conf.algo.pretrained_rep}_freeze_{conf.algo.freeze_pretrained_rep}_r2p_{conf.rep_to_policy}_seed_{conf.seed}"

    log.info(f"Running experiment with config:\n{OmegaConf.to_yaml(conf)}\n")
    log.info(f"Saving experiment logs in {exp_dir}")

    # Save conf
    torch.save(conf, f=f"{exp_dir}/conf.pth")

    # Load train and test trajectories
    with open(conf.task.demo_path, "r") as f:
        traj_info = json.load(f)
    train_traj_stats = traj_info["train_demo_stats"]
    test_traj_stats = traj_info["test_demo_stats"]

    pretrained_rep = conf.algo.pretrained_rep

    demo_root_dir = os.path.join(os.path.dirname(conf.task.demo_path), os.pardir)
    # Get traj lists (read from demo files) and add to traj_info
    traj_info["train_demos"] = t_utils.get_traj_list(demo_root_dir, train_traj_stats, "pos")
    traj_info["test_demos"] = t_utils.get_traj_list(demo_root_dir, test_traj_stats, "pos")

    if not conf.no_wandb:
        wandb_info = t_utils.configure_wandb(exp_id, exp_dir, conf)
        log.info(f"Start wandb logging with info\n{wandb_info}")


    bc = BCFinetune(conf, traj_info, device)
    bc.train(model_data_dir=exp_dir, no_wandb=conf.no_wandb)


if __name__ == "__main__":
    main()
