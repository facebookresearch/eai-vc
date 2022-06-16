#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python src/eval_multitask.py \
    task=cheetah-mt9 \
    algorithm=tdmpc \
    exp_name=first-b4096
