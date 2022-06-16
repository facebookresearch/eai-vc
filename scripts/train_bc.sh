#!/bin/bash
export WANDB_API_KEY=c022235fabdd5619dc99dd0ac117d9e6aabaf8fd
CUDA_VISIBLE_DEVICES=4 python src/train_offline.py \
    task=walker-mt3 \
    algorithm=bc \
    expert_actions=True \
    exp_name=bc-expert
