import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import wandb
import pandas as pd
from pathlib import Path
from logger import make_dir
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
DEFAULT_PLOT_RC = {
    "axes.labelsize": 14,
    "axes.titlesize": 20,
    "axes.facecolor": "#F6F6F6",
    "axes.edgecolor": "#333",
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "xtick.bottom": True,
    "xtick.color": "#333",
    "ytick.labelsize": 16,
    "ytick.left": True,
    "ytick.color": "#333",
}
sns.set(style="whitegrid", rc=DEFAULT_PLOT_RC)
ENTITY = "nihansen"
PROJECT = "tdmpc2"


def main():
    tasks = [
        "mw-mt15",
        "mw-drawer-close",
        "mw-drawer-open",
        "mw-hammer",
        "mw-box-close",
        "mw-reach",
        "mw-push",
        "mw-pick-place",
        "mw-assembly",
        "mw-soccer",
        "mw-faucet-close",
        "mw-faucet-open",
        "mw-door-open",
        "mw-door-close",
        "mw-window-open",
        "mw-window-close",
    ]
    exp_names = ["offline-v1-again-per", "offline-v1-taskenc-b2048-e1024"]
    experiment2label = {
        "state-offline-v1-again-per": "State",
        "state-offline-v1-taskenc-b2048-e1024": "State MT15",
        "pixels-offline-v1-again-per": "Pixels",
    }
    num_seeds = 3
    seeds = set(range(1, num_seeds + 1))

    api = wandb.Api(timeout=100)
    runs = api.runs(
        os.path.join(ENTITY, PROJECT),
        filters={
            "$or": [{"tags": task} for task in tasks],
            "$or": [{"tags": f"seed:{s}"} for s in seeds],
            "$or": [{"tags": exp_name} for exp_name in exp_names],
        },
    )
    print(f"Found {len(runs)} runs after filtering")

    entries = []
    for run in runs:
        cfg = {k: v for k, v in run.config.items()}
        try:
            seed = int(run.name)
        except:
            continue
        task = cfg.get("task", None)
        exp_name = cfg.get("exp_name", None)
        if task not in tasks or exp_name not in exp_names or seed not in seeds:
            continue
        if task == "mw-mt15":
            keys = [f"offline/task_reward/{task}" for task in tasks[1:]]
        else:
            keys = ["offline/reward"]
        hist = run.history(keys=keys, x_axis="_step")
        if len(hist) < 6:
            continue
        experiment = cfg["modality"] + "-" + exp_name
        label = experiment2label[experiment]
        idx = list(experiment2label.values()).index(label)

        if task == "mw-mt15":
            print(f"Appending experiment {experiment}")
            for task in tasks[1:]:
                reward = np.array(hist[f"offline/task_reward/{task}"])[-1]
                entries.append((idx, task, label, seed, reward))
        else:
            reward = np.array(hist[keys[0]])[-1]
            print(f"Appending experiment {experiment} with reward {reward}")
            entries.append((idx, cfg["task"], label, seed, reward))

    df = pd.DataFrame(entries, columns=["idx", "task", "experiment", "seed", "reward"])

    # average across tasks
    df = df.groupby(["idx", "experiment", "seed"]).mean().reset_index()

    # print unique experiments
    print(df["experiment"].unique())

    # average across seeds
    df = df.groupby(["idx", "experiment"]).mean().reset_index()

    # rescale reward
    df["reward"] = (df["reward"] / 45).round()

    f, ax = plt.subplots(1, 1, figsize=(8, 6))

    # metaworld
    sns.barplot(data=df, x="experiment", y="reward", ax=ax, ci=None)
    ax.set_title("Meta-World MT15", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.set_ylabel("Normalized return")
    ax.bar_label(ax.containers[0], fontsize=18)
    ax.tick_params(labelrotation=35)
    for i in range(1, len(df), 2):
        ax.containers[0].patches[i]._hatch = "."
        ax.containers[0].patches[i].set_facecolor(
            ax.containers[0].patches[i - 1]._facecolor
        )

    h, l = ax.get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=4, frameon=False)
    plt.tight_layout()
    f.subplots_adjust(bottom=0.16, wspace=0.15)
    plt.savefig(Path(make_dir("plots")) / "multitask.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
