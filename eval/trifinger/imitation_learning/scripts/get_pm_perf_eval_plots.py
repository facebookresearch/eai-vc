import argparse
import os
import os.path as osp

import hydra
import numpy as np
import pandas as pd
from imitation_learning.run import main as imitation_run
from omegaconf import OmegaConf
from rl_utils.plotting.auto_table import MISSING_VALUE, plot_table
from rl_utils.plotting.wb_query import query


@hydra.main(config_path="pm_plots", config_name="pm")
def main(cfg):
    lookup_k = "dist_to_goal"

    proj_cfg = OmegaConf.load(osp.join(osp.expanduser("~"), "configs/mbirlo.yaml"))

    os.makedirs(cfg.output_dir, exist_ok=True)
    for method, name in cfg.render_names.items():
        model_info = query(["last_model", "config"], {"id": name}, proj_cfg)
        model_info = model_info[0]
        config = model_info["config"]
        hack_replace(lambda: config["policy_updater"]["reward"])
        hack_replace(lambda: config["policy_updater"])
        hack_replace(lambda: config["policy_updater"]["inner_updater"])
        hack_replace(lambda: config["policy_updater"]["discriminator"])
        hack_replace(lambda: config["policy_updater"]["policy_updater"])
        hack_replace(lambda: config["evaluator"])

        eval_cfg = OmegaConf.create(config)
        eval_cfg.num_eval_episodes = 20
        eval_cfg.logger._target_ = "rl_utils.logging.Logger"
        eval_cfg.logger.vid_dir = cfg.output_dir
        eval_cfg.logger.run_name = method
        eval_cfg.load_checkpoint = model_info["last_model"]
        eval_cfg.load_policy = True
        eval_cfg.only_eval = True
        eval_cfg.evaluator.plt_lim = 1.15
        eval_cfg.evaluator.is_final_render = True
        eval_cfg.evaluator.plt_density = 120
        eval_cfg.evaluator.with_arrows = cfg.with_arrows
        if "Obstacle" not in eval_cfg.env.env_name:
            eval_cfg.env.env_settings.params.random_start_region_sample = False

        imitation_run(eval_cfg)


def hack_replace(get_k):
    try:
        v = get_k()["_target_"]
        if not v.startswith("imitation_learning."):
            get_k()["_target_"] = f"imitation_learning.{v}"
    except:
        pass


if __name__ == "__main__":
    main()
