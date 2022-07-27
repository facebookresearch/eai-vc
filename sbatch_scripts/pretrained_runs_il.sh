#!/bin/bash
source ./sbatch_scripts/training_function_il.sh

## Slurm
REPO_NAME="mae-for-eai"
MAIN_USER="karmeshyadav"
REPO_PATH="/private/home/${MAIN_USER}/mae/${REPO_NAME}"
PARTITION="learnfair,learnlab,devlab"
SPLIT="train"
VAL_SPLIT="val"
BASE_TASK_CONFIG_PATH="${REPO_PATH}/configs/tasks/objectnav_hm3d_il.yaml"
EXP_CONFIG_PATH="${REPO_PATH}/configs/experiments/il_ddp_objectnav.yaml"
NODES=4
WANDB_MODE="online"
ENVIRONMENT="hm3d"
VIDEO_OPTION="[]"
CKPT_INTERVAL=1000
NUM_UPDATES=20000
TIME="72:00:00"
NUM_ENV=6
TEST_EPISODE_COUNT=4200
RUN_TRAIN_SCRIPT=true
RUN_EVAL_SCRIPT=false

# EXP_NAME="OVRL_new_reward"
# WEIGHTS_NAME="omnidata_DINO_02.pth"
# BACKBONE="resnet50"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs True \
#             RL.PPO.lr 2.5e-4"
# run_training 0

EXP_NAME="mae_vit_small_decoder_large_HGPS_RE10K_100_first_expt"
WEIGHTS_NAME="mae_vit_small_decoder_large_HGPS_RE10K_100.pth"
BACKBONE="vit_small_patch16"
EXTRA_CMDS="MODEL.RGB_ENCODER.freeze_backbone False \
            MODEL.RGB_ENCODER.randomize_augmentations_over_envs False \
            MODEL.STATE_ENCODER.rnn_type GRU \
            IL.BehaviorCloning.lr 1e-3"
NODES=8
run_training 0

