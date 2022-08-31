Utility script to launch jobs on Slurm (either via sbatch or srun), in a new tmux window, with PyTorch distributed, or in the current shell.

## Run Exp Launcher

Keys in config file.
* `add_all: str`: Suffix that is added to every command.
* `add_all: List[str]`: List of Slurm hosts that should be ignored.

Variables that are automatically substituted into the commands:
* `$GROUP_ID`: A unique generated ID assigned to all runs from the command.
