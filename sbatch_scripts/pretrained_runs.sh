#!/bin/bash
source ./sbatch_scripts/training_function.sh

## Slurm
REPO_NAME="mae-for-eai"
MAIN_USER="karmeshyadav"
REPO_PATH="/private/home/${MAIN_USER}/mae/${REPO_NAME}"
PARTITION="learnfair,learnlab,devlab"
SPLIT="train"
VAL_SPLIT="val_all"
BASE_TASK_CONFIG_PATH="${REPO_PATH}/configs/tasks/imagenav.yaml"
EXP_CONFIG_PATH="${REPO_PATH}/configs/experiments/imagenav.yaml"
NODES=4
WANDB_MODE="online"
ENVIRONMENT="gibson"
VIDEO_OPTION="[]"
NUM_STEPS=5e8
TIME="72:00:00"
NUM_ENV=10
TEST_EPISODE_COUNT=4200
RUN_TRAIN_SCRIPT=true
RUN_EVAL_SCRIPT=false

# EXP_NAME="DINO_reproduce"
# WEIGHTS_NAME="omnidata_DINO_02.pth"
# BACKBONE="resnet50"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="mae_frozen_first_experiment"
# WEIGHTS_NAME="osd_1_45m_mae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True"
# SPLIT="train_extra"
# run_training 0

EXP_NAME="mae_finetuned_first_experiment"
WEIGHTS_NAME="osd_1_45m_mae_base_01.pth"
BACKBONE="vit_base_patch16"
EXTRA_CMDS="RL.POLICY.freeze_backbone False"
SPLIT="train_extra"
NUM_ENV=6
run_training 0