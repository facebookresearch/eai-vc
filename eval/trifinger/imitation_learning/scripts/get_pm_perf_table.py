"""
Run this locally
"""
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
    lookup_k = "eval.dist_to_goal"

    proj_cfg = OmegaConf.load(osp.join(osp.expanduser("~"), "configs/mbirlo.yaml"))

    data = []
    for mode, methods in cfg.methods.items():
        for method_name, group in methods.items():
            r = []
            if len(group) != 0:
                r = query([lookup_k], {"group": group}, proj_cfg)

            if len(r) == 0:
                r = [{lookup_k: MISSING_VALUE}]
            for d in r:
                data.append({"method": method_name, "type": mode, "dist": d[lookup_k]})
    total_df = pd.DataFrame(data)

    def highlight(row_k, row):
        if "train" in row_k:
            return None
        return row.index[row.argmin()]

    plot_table(
        total_df,
        "method",
        "type",
        "dist",
        col_order=["mirl", "airl", "gcl", "maxent"],
        renames={
            "train": "Train",
            "train_eval": "Eval (Train)",
            "eval": "Eval (Test)",
            "mirl": "Meta-IRL",
            "airl": "AIRL",
            "gcl": "GCL",
            "maxent": "MaxEnt",
        },
        row_order=["train", "train_eval", "eval"],
        get_row_highlight=highlight,
        write_to=cfg.write_to,
    )


if __name__ == "__main__":
    main()
