# RL Utilities
A library of helper functions, environment helpers, and experiment management functions for RL research.
* `envs` utility to setup vectorized environments.
    * `envs/pointmass` A toy navigation task implemented in PyTorch.
* `logging` run CLI, Wandb, or Tensorboard logging.
* `models` useful model components for RL policy networks.
* `common`  helpers to manipulate observation and action spaces, standardize policy evaluation, and help with visualizing policy rollouts.

# Installation
Requires Python >= 3.7.

Install from source for development:
* Clone this repository `git clone https://github.com/ASzot/rl-utils.git`
* `pip install -e .`

# Environments
* [Point Mass Navigation](https://github.com/ASzot/rl-helper/tree/main/rl_utils/envs/pointmass): A 2D point navigates to the goal.
