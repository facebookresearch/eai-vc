#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a launcher script for launching mjrl training using hydra
"""

import os
import hydra
import multiprocessing
from omegaconf import OmegaConf

os.environ["MUJOCO_GL"] = "egl"

cwd = os.getcwd()


# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_path="config", config_name="DMC_BC_config", version_base="1.1")
def configure_jobs(config: dict) -> None:
    print("========================================")
    print("Job Configuration")
    print("========================================")

    config = OmegaConf.structured(OmegaConf.to_yaml(config))

    from train_loop import bc_pvr_train_loop

    config["cwd"] = cwd
    with open("job_config.json", "w") as fp:
        OmegaConf.save(config=config, f=fp.name)
    print(OmegaConf.to_yaml(config))

    bc_pvr_train_loop(config)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    configure_jobs()
