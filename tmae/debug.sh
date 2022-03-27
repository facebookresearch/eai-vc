#!/bin/bash

NGPUS=$(nvidia-smi --list-gpus | wc -l)

DATA="data/datasets/hm3d+gibson/v1"

set -x
torchrun \
--nproc_per_node $NGPUS \
main_pretrain.py \
--batch_size 128 \
--epochs 800 \
--accum_iter 4 \
--model mae_vit_base_patch16 \
--max_offset 16 \
--mask_ratio1 0.75 \
--mask_ratio2 0.95 \
--loss_weight 0.5 \
--norm_pix_loss \
--weight_decay 0.05 \
--blr 1.5e-4 \
--warmup_epochs 40 \
--data_path $DATA \
