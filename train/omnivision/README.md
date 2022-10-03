## Omnivision Trainer [![CircleCI](https://circleci.com/gh/fairinternal/omnivision.svg?style=svg&circle-token=7b992a7b384a827bbe71b0fe9d49f71909f6f044)](https://circleci.com/gh/fairinternal/omnivision)

## Working with RSC

Follow the specific instructions listed in the [README_RSC.md](README_RSC.md).

## Installation

Omnivision requires Python 3.9. To install PyTorch 1.12 with CUDA 11.3 on Python 3.9 via conda, run the following instructions -

```bash
conda create --name ov python=3.9
conda activate ov
conda install pytorch=1.12 torchvision=0.13 torchaudio=0.12 cudatoolkit=11.3 -c pytorch
```

Install Omnivision in developer mode (where it runs code from your local checkout and picks up your changes) by running -
```bash
git clone https://github.com/fairinternal/omnivision.git
cd omnivision

pip install -e ".[dev,<YOUR PROJECT>]"
# For Omnivore this would be,
pip install -e ".[dev,omnivore]"
# And for Omniscale this would be,
pip install -e ".[dev,omniscale]"
```

Next, you will be working on a project within omnivision. To install all the projects, run -
```bash
pip install -e projects
```

## FBCode: On Demand GPUs

We rely on conda to launch jobs, even on the FB cluster. To persist your conda environments,
create and activate them from your devserver's storage.
You will need to enable devserver mounting via:

`bunny ondemand` -> `fbsource` -> `Edit On Demand Preferences` -> `Storage to mount`

Note that since packages are downloaded from the internet, you will have to run
such commands using `$ with-proxy $CMD`.

To use conda, follow these steps -

```bash
DEVSERVER=devbig016.atn6.facebook.com  # replace with your devserver
# mounts your devserver if it isn't already mounted
# we will be saving the conda environment to your primary devserver since
# ondemands don't have persistent storage
# this mounts your devserver's ~/local to ~/$DEVSERVER
mount-devserver


# if you don't have a conda installed on your devserver,
# install it (one time)
~/fbcode/caffe2/torch/fb/scripts/setup_ondemand.sh
# move the installed conda to your devserver
mv ~/local/miniconda3 ~/$DEVSERVER/miniconda3


# now we assume there is a conda present on your devserver
ln -s ~/$DEVSERVER/miniconda3 ~/local/miniconda3
source ~/.bashrc
conda env list


# if you haven't set up your conda env, create a new env following the Installation instructions!


# now let's launch a job!
conda activate ov
cd fbcode/deeplearning/projects/omnivision/projects/omnivore/
./dev/launch_job.py -c config/experiments/reference/dummy_kinetics_train_slurm_gpu.yaml --local
```

## Testing

### OSS

Before running the tests, please ensure that you installed the necessary additional test dependencies.

Use the the following command to run the tests:
```bash
# run all tests
python -m unittest discover -v -s tests -t .
# run a specific test
python -m unittest tests.test_scheduler
```

### FB cluster

```bash
# run all the tests
buck test @mode/dev-nosan //deeplearning/projects/omnivision:
# run a specific test
buck test @mode/dev-nosan //deeplearning/projects/omnivision:test_scheduler
```

## Formatting

### OSS

We use ufmt to handle formatting
```bash
ufmt --help
ufmt format
```

### FB cluster

```bash
arc lint
```

## TODOs

### Generic
- [ ] Decide if we should allow unused param names or classes
- [ ] Resolve `NCCL_ASYNC_ERROR_HANDLING` so our distributed ops can timeout - https://fb.workplace.com/groups/319878845696681/permalink/273716954739519/
- [ ] Add `frozen_param_names` and `frozen_module_cls_names` for freezing part of the model
- [ ] Remove dataloader worker logic from SharedNumpyArray.
- [ ] Add memcache to all datasets
- [ ] Support async in OSS
- [ ] Fix resumptions when using layer decay
- [ ] Enable seperate seed setting for each data worker.
- [ ] The `layer_decay_param_modifier` creates a new parameter group for each parameter (hundreds in some cases). Group the parameters with same LR schedule so it's more managable. Also the tensorboard visualizations are messy with 100s of curves for LR schedule.

### FB Cluster
- [ ] Remove opt-split-dwarf from configerator
- [ ] Move to fbpkg builder when it's available for jetter
- [ ] Figure out package expiry on fb side
- [ ] Migrate to fbpkg.builder for jetter bento kernels once available
- [ ] Copy async logic from old omnivore as per D34840826
