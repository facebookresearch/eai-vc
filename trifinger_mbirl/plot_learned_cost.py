import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import os.path
import sys
import torch

def main(args):
    
    data = torch.load(args.file_path)
    conf = data["conf"]
    cost_type = conf.cost_type
    
    cost_weights = data["cost_parameters"]["weights"].detach().numpy()


    d_list = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",]

    if args.save:
        exp_dir = os.path.split(args.file_path)[0]
        save_path = os.path.join(exp_dir, "learned_cost_weights.png")
    else:
        save_path = None

    plot_MPTimeDep("Learned cost weights", cost_weights, d_list, save_path=save_path)

def plot_MPTimeDep(title, cost_weights, d_list, save_path=None):
    """ Plot multi-phase time dependent cost weights """
    
    time, mode, dim = cost_weights.shape

    plt.figure(figsize=(10, 10), dpi=200)
    plt.subplots_adjust(hspace=1)
    plt.suptitle(title)

    k = 0
    offsets = [-0.2, 0.2]
    colors = ["r", "b"]
    for i, d in enumerate(d_list):
        k += 1
        plt.subplot(len(d_list), 1, k)
        if len(d_list) > 1:
            plt.title(f"{d}")

        for m in range(mode):
            plt.bar(np.arange(time) + offsets[m], cost_weights[:, m, i], width=0.4, label=f"Mode {m}", color=colors[m])

    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", default=None, help="""Filepath of log.pth file""")
    parser.add_argument("--save", "-s", action="store_true", help="Save figs")
    args = parser.parse_args()
    main(args)
