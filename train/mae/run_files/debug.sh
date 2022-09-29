#!/bin/bash

NGPUS=$(nvidia-smi --list-gpus | wc -l)

DATA="/checkpoint/karmeshyadav/hm3d+gibson/v1/train"

set -x
torchrun \
--nproc_per_node $NGPUS \
main_pretrain.py \
--batch_size 64 \
--epochs 800 \
--accum_iter 4 \
--model mae_vit_small_patch16 \
--mask_ratio 0.75 \
--norm_pix_loss \
--weight_decay 0.05 \
--blr 1.5e-4 \
--warmup_epochs 40 \
--data_path $DATA \
--wandb_name "debug-mae" \
--wandb_mode "disabled" \
--output_dir /checkpoint/karmeshyadav/mae_training/ \
--color_jitter