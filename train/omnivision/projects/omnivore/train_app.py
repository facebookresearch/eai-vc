# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

import hydra
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf
from omnivore.train_utils import (
    get_machine_local_and_dist_rank,
    makedir,
    register_omegaconf_resolvers,
)


@hydra.main(config_path="config", config_name=None)
def main(cfg) -> None:
    makedir(cfg.launcher.experiment_log_dir)
    _, rank = get_machine_local_and_dist_rank()
    if rank is None or rank == 0:
        with g_pathmgr.open(
            os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
        ) as f:
            f.write(OmegaConf.to_yaml(cfg))

        with g_pathmgr.open(
            os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
        ) as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


if __name__ == "__main__":
    register_omegaconf_resolvers()
    main()
