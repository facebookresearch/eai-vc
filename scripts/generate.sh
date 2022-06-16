#!/bin/bash
export WANDB_API_KEY=c022235fabdd5619dc99dd0ac117d9e6aabaf8fd
declare -a TASKS=( \
    "walker-walk"
)
NUM_TASKS=${#TASKS[@]}

echo "Running ${NUM_TASKS} task(s)..."

for (( i=0; i<${NUM_TASKS}; i++ )); do

        CUDA_VISIBLE_DEVICES=0 python src/generate.py \
            task=${TASKS[$i]} \
            algorithm=tdmpc \
            exp_name=v1 \
            eval_freq=50000 \
            eval_episodes=30 \
            seed=1,2,3

    done
done
wait
