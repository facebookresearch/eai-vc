#!/bin/bash

NGPUS=$(nvidia-smi --list-gpus | wc -l)

DATA="data/datasets/imagenet/train"

set -x
torchrun \
--nproc_per_node $NGPUS \
main_pretrain.py \
--batch_size 128 \
--epochs 800 \
--accum_iter 4 \
--model mae_vit_base_patch16 \
--mask_ratio 0.75 \
--norm_pix_loss \
--weight_decay 0.05 \
--blr 1.5e-4 \
--warmup_epochs 40 \
--data_path $DATA \
--wandb_name "debug" \
--wandb_mode "online" \
