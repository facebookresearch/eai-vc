import os.path as osp

import matplotlib.pyplot as plt
import numpy as np


def plot_actions(pred_actions, gt_actions, n_updates, save_dir):
    assert pred_actions.shape == gt_actions.shape
    action_len, action_dim = pred_actions.shape

    for action_dim_i in range(action_dim):
        plt.scatter(
            np.arange(action_len), pred_actions[:, action_dim_i], label="Predicted"
        )
        plt.scatter(np.arange(action_len), gt_actions[:, action_dim_i], label="Expert")
        plt.legend()
        plt.title(f"Action Update Batch Dim {action_dim_i} @ {n_updates}")
        plt.savefig(osp.join(save_dir, f"actions_{n_updates}_{action_dim_i}.png"))
        plt.clf()
