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

EXP_NAME="pretrained_omnidata3_6m_MOCO_01_finetuned_01"
WEIGHTS_NAME="omnidata3_6m_MOCO_01.pth.tar"
BACKBONE="resnet50_gn"
EXTRA_CMDS="RL.DDPPO.pretrained_encoder True \
            RL.DDPPO.freeze_backbone True"
SPLIT="train_extra"
run_training 0
