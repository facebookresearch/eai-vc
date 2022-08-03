#!/bin/bash
source ./sbatch_scripts/training_function.sh

## Slurm
REPO_NAME="mae-for-eai"
MAIN_USER=${USER}
REPO_PATH="/private/home/${MAIN_USER}/${REPO_NAME}"
PARTITION="learnfair,learnlab,devlab"
SPLIT="train_extra"
VAL_SPLIT="val_all"
SEED=1
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

RUN_TRAIN_SCRIPT=false
RUN_EVAL_SCRIPT=false
if [[ $2 == "train" ]]; then
    RUN_TRAIN_SCRIPT=true
elif [[ $2 == "eval" ]]; then
    RUN_EVAL_SCRIPT=true
else
    echo "Invalid training mode"
    exit 1
fi

echo $1 $2

# Training from Scratch
if [[ $1 = "scratch" ]]; then
    EXP_NAME="Scratch_First_Run"
    WEIGHTS_NAME=""
    BACKBONE="resnet50"
    EXTRA_CMDS="RL.POLICY.freeze_backbone False \
                RL.POLICY.rnn_type LSTM \
                RL.POLICY.randomize_augmentations_over_envs True \
                RL.PPO.lr 2.5e-4"
    run_training ${SEED}
fi

# Training Dino
if [[ $1 = "dino" ]]; then
    EXP_NAME="DINO_First_Run"
    WEIGHTS_NAME="omnidata_DINO_02.pth"
    BACKBONE="resnet50"
    EXTRA_CMDS="RL.POLICY.freeze_backbone False \
                RL.POLICY.rnn_type LSTM \
                RL.POLICY.randomize_augmentations_over_envs True \
                RL.PPO.lr 2.5e-4"
    run_training ${SEED}
fi

# Training MAE/TMAE
if [[ $1 = "mae" ]]; then
    EXP_NAME="MAE_First_Run"
    WEIGHTS_NAME="mae_vit_small_decoder_large_HGPS_RE10K_100.pth"
    BACKBONE="vit_small_patch16"
    EXTRA_CMDS="RL.POLICY.freeze_backbone False \
                RL.POLICY.rnn_type LSTM \
                RL.POLICY.randomize_augmentations_over_envs False \
                RL.PPO.lr 6.25e-5 \
                RL.POLICY.vit_global_pool False \
                RL.POLICY.vit_use_fc_norm False"
    NUM_ENV=8
    NODES=5
    run_training ${SEED}
fi

# Training Data2Vec
if [[ $1 = "d2v" ]]; then
    EXP_NAME="Data2Vec_First_Run"
    WEIGHTS_NAME="data2vec_128bsz_top3_e400_n2_399.pth"
    BACKBONE="beit_base_patch16"
    EXTRA_CMDS="RL.POLICY.freeze_backbone False \
                RL.POLICY.rnn_type LSTM \
                RL.POLICY.randomize_augmentations_over_envs False \
                RL.PPO.lr 6.25e-5 \
                RL.POLICY.vit_global_pool False \
                RL.POLICY.vit_use_fc_norm False"
    NUM_ENV=4
    NODES=10
    run_training ${SEED}
fi

