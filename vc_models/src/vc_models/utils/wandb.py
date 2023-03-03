#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import omegaconf


log = logging.getLogger(__name__)


def setup_wandb(config):
    try:
        log.info(f"wandb initializing...")
        import wandb

        wandb_run = start_wandb(config, wandb)

        log.info(f"wandb initialized")

        return wandb_run
    except Exception as e:
        log.warning(f"Cannot initialize wandb: {e}")
        return


def start_wandb(config, wandb):
    resume = "allow"
    wandb_id = wandb.util.generate_id()

    if "dir" in config.wandb and config.wandb.dir is not None:
        wandb_filename = os.path.join(config.wandb.dir, "wandb", "wandb_id.txt")
        if os.path.exists(wandb_filename):
            # if file exists, then we are resuming from a previous eval
            with open(wandb_filename, "r") as file:
                wandb_id = file.read().rstrip("\n")
            resume = "must"
        else:
            os.makedirs(os.path.dirname(wandb_filename), exist_ok=True)
            with open(wandb_filename, "w") as file:
                file.write(wandb_id)

    if isinstance(config, omegaconf.DictConfig):
        config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
    wandb_cfg_dict = config["wandb"]

    return wandb.init(id=wandb_id, config=config, resume=resume, **wandb_cfg_dict)
