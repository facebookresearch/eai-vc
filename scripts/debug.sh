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
