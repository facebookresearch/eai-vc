#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from datetime import datetime
import os
import subprocess

import hydra
from omegaconf import DictConfig
import numpy as np
import torch

from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)
from habitat.config.default_structured_configs import HabitatConfigPlugin
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.ddp_utils import rank0_only
from habitat2_vc.policy import EAIPolicy  # noqa: F401


def get_random_seed():
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    print("Using a generated random seed {}".format(seed))
    return seed


def setup_experiment(config: DictConfig):
    """
    Setups the random seed and the wandb logger.
    """
    # Set random seed
    seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Add the seed to the config
    config.habitat.seed = seed

    # Single-agent setup
    config.habitat.simulator.agents_order = list(config.habitat.simulator.agents.keys())

    # Add the wandb information to the habitat config
    config.habitat_baselines.wb.project_name = config.WANDB.project
    config.habitat_baselines.wb.run_name = config.WANDB.name
    config.habitat_baselines.wb.group = config.WANDB.group
    config.habitat_baselines.wb.entity = config.WANDB.entity

    # Set torch to single threaded
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    # Create the checkpoint and video directories
    if rank0_only():
        os.makedirs(config.habitat_baselines.checkpoint_folder, exist_ok=True)
        os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

    # Create the symlink to the data folder
    data_path = hydra.utils.to_absolute_path(config.habitat.dataset.data_path)
    base_data_path = data_path.split("data/")[0] + "data/"

    subprocess.call(
        [
            "ln",
            "-s",
            base_data_path,
            "data",
        ]
    )

    # Set the log levels
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(config: DictConfig) -> None:
    r"""Main function for habitat_vc
    Args:
        cfg: DictConfig object containing the configs for the experiment.
    """
    # Setup the experiment
    setup_experiment(config)

    # Get the trainer
    trainer_init = baseline_registry.get_trainer(config.habitat_baselines.trainer_name)
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"

    trainer = trainer_init(config)

    # Train or eval
    if config.RUN_TYPE == "train":
        trainer.train()
    elif config.RUN_TYPE == "eval":
        trainer.eval()


if __name__ == "__main__":
    # Register habitat hydra plugins
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    register_hydra_plugin(HabitatConfigPlugin)
    # Call hydra main
    main()
