#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

NGPUS=$(nvidia-smi --list-gpus | wc -l)
ROOT_DIR=$(git rev-parse --show-toplevel)

set -x
torchrun \
--standalone \
--nnodes 1 \
--nproc_per_node 1 \
run.py \
RUN_TYPE=train \
LOG_INTERVAL=1 \
NUM_ENVIRONMENTS=4 \
RL.POLICY.freeze_backbone=True \
TASK_CONFIG.DATASET.SCENES_DIR=data/scene_datasets \
TASK_CONFIG.DATASET.DATA_PATH=data/datasets/pointnav/gibson/v1/train_extra/train_extra.json.gz \
WANDB.mode="disabled" \
model=mae_small_HGSP_RE10K_100

