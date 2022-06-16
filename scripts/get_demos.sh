#!/bin/bash
export WANDB_API_KEY=c022235fabdd5619dc99dd0ac117d9e6aabaf8fd
declare -a GPUS=(0)
declare -a TASKS=( \
    # "acrobot-swingup" \
    # "cartpole-balance" \
    # "cartpole-swingup" \
    # "cartpole-two-poles" \
    # "cheetah-run" \
    # "cup-catch" \
    # "dog-run" \
    # "dog-trot" \
    # "dog-walk" \
    # "finger-spin" \
    # "finger-turn-easy" \
    # "finger-turn-hard" \
    # "fish-swim" \
    # "hopper-hop" \
    # "hopper-stand" \
    # "humanoid-run" \
    # "humanoid-stand" \
    # "humanoid-walk" \
    # "pendulum-swingup" \
    # "quadruped-run" \
    # "quadruped-walk" \
    # "reacher-easy" \
    # "reacher-hard" \
    # "walker-run" \
    # "walker-stand" \
    # "walker-walk" \
    "walker-flip"
)
NUM_GPUS=${#GPUS[@]}
NUM_TASKS=${#TASKS[@]}
MAX_ITERATIONS=1

echo "Running ${NUM_TASKS} task(s) on ${NUM_GPUS} GPU(s)..."

for (( i=0; i<${NUM_TASKS}; i++ )); do
    for ((j=0; j<=${MAX_ITERATIONS}; j++)); do

        # CUDA_VISIBLE_DEVICES=${GPUS[${i%NUM_GPUS}]} python src/get_demos.py \
        CUDA_VISIBLE_DEVICES=0 python src/get_demos.py \
            task=${TASKS[$i]} \
            action_repeat=2 \
            exp_name=demo \
            eval_episodes=30 \
            demo_iterations=${j} \
            seed=1 #,2,3 # &

    done
done
wait
