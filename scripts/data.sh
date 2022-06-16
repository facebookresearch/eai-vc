#!/bin/bash
export WANDB_API_KEY=c022235fabdd5619dc99dd0ac117d9e6aabaf8fd

CUDA_VISIBLE_DEVICES=0 python src/generate.py \
    task="walker-walk" \
    algorithm=tdmpc \
    exp_name=v1 \
    eval_freq=50000 \
    eval_episodes=30 \
    seed=1
