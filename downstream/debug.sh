#!/bin/bash

NGPUS=$(nvidia-smi --list-gpus | wc -l)

DATA="data/datasets/places365_standard"
CKPT="../data/models/mae_pretrain_vit_base_full.pth"

set -x
torchrun \
--nproc_per_node $NGPUS \
main_linprobe.py \
--batch_size 2048 \
--model vit_base_patch16 \
--cls_token \
--finetune ${CKPT} \
--epochs 90 \
--blr 0.1 \
--weight_decay 0.0 \
--dist_eval \
--num_workers 6 \
--data_path $DATA \
--dataset "places-indoor" \
