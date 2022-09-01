#!/usr/bin/env python3
import argparse
import os
import random
from datetime import datetime

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

import habitat
from habitat import logger
from habitat.config import Config
from habitat.config.default import Config as CN
from habitat_baselines.common.baseline_registry import baseline_registry

from algorithm.config import get_config


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    r"""Main function for habitat_eaif
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
    logger.info("Using a generated random seed {}".format(seed))
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
    cfg = OmegaConf.to_container(cfg)
    cfg = CN(cfg)

    config = get_config()
    config.merge_from_other_cfg(cfg)
    execute_exp(config)


if __name__ == "__main__":
    main()
