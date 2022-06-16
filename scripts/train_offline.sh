#!/bin/bash
export WANDB_API_KEY=c022235fabdd5619dc99dd0ac117d9e6aabaf8fd
CUDA_VISIBLE_DEVICES=6 python src/train_offline.py \
    task=cheetah-mt9 \
    algorithm=bc \
    batch_size=4096 \
    eval_freq=25000 \
    exp_name=first-b4096
