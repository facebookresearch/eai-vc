#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
import os
import random
from datetime import datetime

import habitat

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from habitat.config import Config
from habitat.config.default import Config as CN
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.ddp_utils import rank0_only
from habitat_vc.config import get_config


@hydra.main(config_path="configs", config_name="config_imagenav")
def main(cfg: DictConfig) -> None:
    r"""Main function for habitat_vc
    Args:
        cfg: DictConfig object containing the configs for the experiment.
    """
    run_exp(cfg)


def execute_exp(config: Config) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    """
    # set a random seed (from detectron2)
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    print("Using a generated random seed {}".format(seed))
    config.defrost()
    if config.RUN_TYPE == "eval":
        config.TASK_CONFIG.TASK.ANGLE_SUCCESS.USE_TRAIN_SUCCESS = False
        config.TASK_CONFIG.TASK.IMAGEGOAL_ROTATION_SENSOR.SAMPLE_ANGLE = False
    config.TASK_CONFIG.SEED = seed
    config.freeze()
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    setup_experiment(config)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if config.RUN_TYPE == "train":
        trainer.train()
    elif config.RUN_TYPE == "eval":
        trainer.eval()


def run_exp(cfg: DictConfig) -> None:
    r"""Runs experiment given mode and config

    Args:
        cfg: DictConfig object containing the configs for the experiment.

    Returns:
        None.
    """
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = CN(cfg)

    config = get_config()
    config.merge_from_other_cfg(cfg)
    execute_exp(config)


def setup_experiment(config: Config) -> None:
    if rank0_only():
        os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)

    config.defrost()
    config.TASK_CONFIG.DATASET.SCENES_DIR = hydra.utils.to_absolute_path(
        config.TASK_CONFIG.DATASET.SCENES_DIR
    )
    config.TASK_CONFIG.DATASET.DATA_PATH = hydra.utils.to_absolute_path(
        config.TASK_CONFIG.DATASET.DATA_PATH
    )
    config.freeze()

    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu/nvidia-opengl:" + os.environ["LD_LIBRARY_PATH"]
    )
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["MAGNUM_LOG"] = "quiet"


if __name__ == "__main__":
    main()
