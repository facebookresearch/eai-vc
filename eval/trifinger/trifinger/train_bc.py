# w Copyright (c) Facebook, Inc. and its affiliates.
import random
import os
import torch
import numpy as np
import wandb
import hydra
from omegaconf import OmegaConf
import logging
import json

import utils.train_utils as t_utils
from algos.bc_finetune import BCFinetune

# A logger for this file
log = logging.getLogger(__name__)

"""
Main script to launch imitation learning training
(mbirl, policy_opt, bc, bc_finetune)
"""


@hydra.main(version_base=None, config_path="config", config_name="bc_default")
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

    # Get traj lists (read from demo files) and add to traj_info
    traj_info["train_demos"] = t_utils.get_traj_list(train_traj_stats, "pos")
    traj_info["test_demos"] = t_utils.get_traj_list(test_traj_stats, "pos")

    if not conf.no_wandb:
        # wandb init
        wandb_info_pth = os.path.join(exp_dir, "wandb_info.pth")
        if os.path.exists(wandb_info_pth):
            # Resume logging to existing wandb run
            wandb_info = torch.load(wandb_info_pth)
            conf_for_wandb = wandb_info["conf_for_wandb"]
            wandb_id = wandb_info["id"]
            exp_str = wandb_info["run_name"]
        else:
            # Convert conf to container, so I can add "exp_id" and conf of forward model
            conf_for_wandb = OmegaConf.to_container(
                conf, resolve=True, throw_on_missing=True
            )
            conf_for_wandb["exp_id"] = exp_id  # Add experiment id to conf, for wandb

            # If using a forward model, add forward_model_ckpt["conf"].algo args to wandb conf
            if (
                "mpc_forward_model_ckpt" in conf_for_wandb["algo"]
                and conf_for_wandb["algo"]["mpc_forward_model_ckpt"]
            ):
                ckpt_info = torch.load(conf_for_wandb["algo"]["mpc_forward_model_ckpt"])
                conf_for_wandb["dyn_model"] = ckpt_info["conf"].copy()
            # Then, convert conf_for_wandb container --> omegaconf --> container
            # Needed to do this to get dyn_model params to log as dyn_model.param in wandb
            conf_for_wandb = OmegaConf.to_container(
                OmegaConf.create(conf_for_wandb), resolve=True, throw_on_missing=True
            )
            wandb_id = wandb.util.generate_id()
            wandb_info = {
                "run_name": exp_id,  # exp_str,
                "id": wandb_id,
                "conf_for_wandb": conf_for_wandb,
            }
            torch.save(wandb_info, wandb_info_pth)

        wandb.init(
            project=conf.algo.name + "_" + conf.run_name,
            entity="fmeier",
            id=wandb_info["id"],
            name=wandb_info["run_name"],
            config=wandb_info["conf_for_wandb"],
            settings=wandb.Settings(start_method="thread"),
            resume="allow",
        )
        log.info(f"Start wandb logging with info\n{wandb_info}")

    bc = BCFinetune(conf, traj_info, device)
    bc.train(model_data_dir=exp_dir, no_wandb=conf.no_wandb)


if __name__ == "__main__":
    main()
