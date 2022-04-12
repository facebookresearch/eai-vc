#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

NGPUS=$(nvidia-smi --list-gpus | wc -l)

set -x
torchrun \
--standalone \
--nnodes 1 \
--nproc_per_node $NGPUS \
run.py \
--exp-config configs/experiments/imagenav.yaml \
--run-type train \
LOG_INTERVAL 1 \
RL.POLICY.backbone vit_small_patch16 \
RL.POLICY.pretrained_encoder data/models/tmae_small_01.pth \
RL.POLICY.freeze_backbone False \
RL.POLICY.vit_use_cls True \
WANDB_MODE "disabled" \
