# EAI Foundations

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/facebookresearch/eai-foundations/tree/main.svg?style=svg&circle-token=52b92efd205cf081e310aaad27be9c74a86190b3)](https://dl.circleci.com/status-badge/redirect/gh/facebookresearch/eai-foundations/tree/main)

This is the FAIR monorepo for investigating large-scale pretraining for embodied AI.

## Installation

See [INSTALLATION.md](INSTALLATION.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) to get started contributing code.

## Directory structure

- `eaif-models`: contains a minimal-dependency pip-installable model loading code, as well as some shared-project utilities.
    - See [README](./eaif-models/README.md) for more details.
- `train`: model training code. Currently composed of a heterogeneous set of repos / environments.
- `eval`: model evaluation code, unified under a single reproducible `environment.yml`.
    - [cifar_lin_probe](./eval/cifar_lin_probe/): a basic unit test check that you're able to load models.
- `data`: Gitignored directory containing (symlinks to) datasets, models, etc.
- `third_party`: Third party submodules which aren't expected to change often.

## Shared resources

**Shared directory**: On FAIR Cluster, we have 20TB allocated to the shared directory `/checkpoint/yixinlin/eaif/`, which includes shared (processed) datasets, third-party libraries, model run results, experiments/sandbox.
Please feel free to use it as the shared directory for this project.

The directory is owned by the group `eaif`.
Add yourself to this group by raising a task for Penguin (example [task](https://www.internalfb.com/tasks/?t=128888137)).
For the group changes to take effect, you may need to log out of all ssh/screen/tmux sessions, and restart the tmux server (by running `tmux kill-server`).

**Shared public wandb**:

We have a (public-cloud) team on Weights and Biases (outside of the FAIR cluster), with the team `eai-foundations`. Add anyone on the team through [this link](https://wandb.ai/eai-foundations/members).

If you were logged into the FAIR instance, you may need to relogin:

```
# switch to public cloud
wandb login --host=https://api.wandb.ai --relogin

# switch to FAIR cluster instance
wandb login --host=https://fairwandb.org/ --relogin
```
