import argparse
import json
import os
import time

import yaml
from omnivore.dev.launch_env import read_txt_file_with_comments


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def find_incomplete_runs(args):
    killed_runs = {}
    if os.path.isfile(args.killed_runs_file):
        print(f"Read killed runs from {args.killed_runs_file}")
        killed_runs = read_txt_file_with_comments(args.killed_runs_file)
        killed_runs = set(killed_runs)
    exp_dirs = os.listdir(args.root_dir)
    exp_dirs.sort()

    incomplete_exps = []
    now_time = time.time()

    for exp_dir in exp_dirs:
        try:
            exp_num = int(exp_dir.split("_")[0])
        except Exception:
            print(f"{exp_dir} cannot be parsed into an int")
            continue
        if exp_num < args.exp_start_idx:
            continue

        full_path = os.path.join(args.root_dir, exp_dir)
        sweep_dirs = os.listdir(full_path)
        sweep_dirs.sort()
        for sweep in sweep_dirs:
            sweep_full_path = os.path.join(full_path, sweep)
            if sweep_full_path in killed_runs:
                continue
            train_stats_file = os.path.join(sweep_full_path, "logs/train_stats.json")
            log_file = os.path.join(sweep_full_path, "logs/log.txt")
            config_resolved = os.path.join(sweep_full_path, "config_resolved.yaml")
            if not os.path.isfile(config_resolved):
                print(
                    f"{bcolors.WARNING} {sweep_full_path}: no resolved config file{bcolors.ENDC}"
                )
                continue
            with open(config_resolved, "r") as fh:
                cfg = yaml.load(fh, yaml.Loader)
            num_train_epochs = cfg["trainer"]["max_epochs"]
            to_check = False
            if not os.path.isfile(log_file):
                print(
                    f"{bcolors.WARNING}{sweep_full_path} has not started {bcolors.ENDC}"
                )
                continue
            if os.path.isfile(train_stats_file):
                with open(train_stats_file, "r") as fh:
                    dt = fh.readlines()
                last_line = json.loads(dt[-1].strip())
                if "Trainer/epoch" not in last_line:
                    print(
                        f"{bcolors.WARNING}{sweep_full_path} is an old run. Cannot check.{bcolors.ENDC}"
                    )
                    continue
                if last_line["Trainer/epoch"] < (num_train_epochs - 1):
                    to_check = True
            else:
                # training may have just started so there is no `json` file
                to_check = True
            if to_check is False:
                continue

            # check when the log file was last updated
            mtime = os.path.getmtime(log_file)
            time_diff = now_time - mtime
            if time_diff >= args.time_thresh_in_secs:
                print(
                    f"{bcolors.FAIL} {sweep_full_path} not updated in {time_diff / args.time_thresh_in_secs} hours{bcolors.ENDC}"
                )

    print("\n".join(incomplete_exps))


def main():
    parser = argparse.ArgumentParser()
    username = os.environ["USER"]
    parser.add_argument(
        "--root_dir", default=f"/fsx-omnivore/{username}/omnivision_omnivore"
    )
    parser.add_argument("--config_root_dir", default=f"config/experiments/{username}")
    parser.add_argument(
        "--killed_runs_file", default=f"/fsx-omnivore/{username}/killed_runs.txt"
    )
    parser.add_argument("--time_thresh_in_secs", default=3600, type=int)
    parser.add_argument("--exp_start_idx", type=int, default=50)
    args = parser.parse_args()
    args.root_dir = os.path.join(args.root_dir, args.config_root_dir)
    find_incomplete_runs(args)


if __name__ == "__main__":
    main()
