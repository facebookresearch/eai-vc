#!/bin/bash

NGPUS=$(nvidia-smi --list-gpus | wc -l)

DATA="data/datasets/hm3d+gibson/v1/train"

set -x
torchrun \
--nproc_per_node $NGPUS \
main_pretrain.py \
--batch_size 256 \
--epochs 400 \
--accum_iter 4 \
--model mae_vit_small_patch16 \
--max_offset 16 \
--mask_ratio1 0.75 \
--mask_ratio2 0.95 \
--loss_weight 0.5 \
--norm_pix_loss \
--weight_decay 0.05 \
--blr 1.5e-4 \
--warmup_epochs 40 \
--data_path $DATA \
--num_workers 7 \
--wandb_name "debug-tmae" \
--wandb_mode "disabled" \
--color_jitter \
