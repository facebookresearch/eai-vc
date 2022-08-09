# EAI Foundations

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/facebookresearch/eai-foundations/tree/main.svg?style=svg&circle-token=52b92efd205cf081e310aaad27be9c74a86190b3)](https://dl.circleci.com/status-badge/redirect/gh/facebookresearch/eai-foundations/tree/main)

This is the FAIR monorepo for investigating large-scale pretraining for embodied AI.

## Directory structure

- `eaif-models`: contains a minimal-dependency pip-installable model loading code, as well as some shared-project utilities.
    - See [README](./eaif-models/README.md) for more details.
- `train`: model training code. Currently composed of a heterogeneous set of repos / environments.
- `eval`: model evaluation code. Currently composed of a heterogeneous set of repos / environments.
    - The goal is to unify the evaluation framework under a single reproducible `environment.yml`.

## Shared resources

**Shared directory**: On FAIR Cluster, we have 20TB allocated to the shared directory `/checkpoint/yixinlin/eaif/`, which includes shared (processed) datasets, third-party libraries, model run results, experiments/sandbox.
Please feel free to use it as the shared directory for this project; the directory is owned by the group `eaif`, which you can add your unixname to by raising a task for Penguin.

**Shared public wandb**:

We have a (public-cloud) team on Weights and Biases (outside of the FAIR cluster), with the team `eai-foundations`. Ping Yixin to be added.

If you were logged into the FAIR instance, you may need to relogin:

```
# switch to public cloud
wandb login --host=https://api.wandb.ai --relogin

# switch to FAIR cluster instance
wandb login --host=https://fairwandb.org/ --relogin
```
