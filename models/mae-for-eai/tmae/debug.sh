#!/bin/bash

NGPUS=$(nvidia-smi --list-gpus | wc -l)

DATA="data/datasets/hm3d+gibson/v1/train"
DATE=`date +"%Y-%m-%d-%H%M"`

set -x
torchrun \
--nproc_per_node $NGPUS \
--rdzv_endpoint localhost:29401 \
main_pretrain.py \
--batch_size 256 \
--epochs 400 \
--accum_iter 4 \
--model mae_vit_small_patch16 \
--max_offset 1 \
--mask_ratio1 0.75 \
--mask_ratio2 0.95 \
--loss_weight 0.5 \
--norm_pix_loss \
--weight_decay 0.05 \
--blr 1.5e-4 \
--warmup_epochs 40 \
--data_path $DATA \
--num_workers 7 \
--wandb_name "debug-tmae-$DATE" \
--output_dir "output_dir_$DATE" \
--randomize_views \
