#!/bin/bash
export WANDB_API_KEY=c022235fabdd5619dc99dd0ac117d9e6aabaf8fd
CUDA_VISIBLE_DEVICES=3 python src/train.py \
    task=cheetah-legs-up \
    algorithm=tdmpc \
    exp_name=test-getup \
    eval_freq=10000 \
    eval_episodes=3 \
    save_video=True \
    seed=1
