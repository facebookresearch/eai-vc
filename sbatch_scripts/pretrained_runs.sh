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

# EXP_NAME="DINO_reproduce_finetuned_LSTM"
# WEIGHTS_NAME="omnidata_DINO_02.pth"
# BACKBONE="resnet50"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="DINO_reproduce_finetuned_LSTM_with_avgpool"
# WEIGHTS_NAME="omnidata_DINO_02.pth"
# BACKBONE="resnet50"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.avgpooled_image True"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="DINO_reproduce_finetuned_LSTM_diverse_augs"
# WEIGHTS_NAME="omnidata_DINO_02.pth"
# BACKBONE="resnet50"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs True"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="DINO_reproduce_finetuned_GRU"
# WEIGHTS_NAME="omnidata_DINO_02.pth"
# BACKBONE="resnet50"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False"
# SPLIT="train_extra"
# run_training 0


# EXP_NAME="mae_frozen_first_experiment"
# WEIGHTS_NAME="osd_1_45m_mae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="mae_frozen_old_masking"
# WEIGHTS_NAME="osd_1_45m_mae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True \
#             RL.POLICY.vit_use_fc_norm False \
#             RL.POLICY.vit_global_pool True"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="mae_frozen_with_compression"
# WEIGHTS_NAME="osd_1_45m_mae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True \
#             RL.PPO.lr 2.5e-4"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="mae_frozen_without_augmentations"
# WEIGHTS_NAME="osd_1_45m_mae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True \
#             RL.POLICY.use_augmentations False \
#             RL.POLICY.use_augmentations_test_time False"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="mae_frozen_with_compression"
# WEIGHTS_NAME="osd_1_45m_mae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True \
#             RL.PPO.lr 2.5e-4"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="mae_finetuned_first_experiment"
# WEIGHTS_NAME="osd_1_45m_mae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False"
# SPLIT="train_extra"
# NUM_ENV=6
# run_training 0

# EXP_NAME="mae_finetuned_LSTM_" #randomised_envs
# WEIGHTS_NAME="osd_1_45m_mae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs True"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="mae_finetuned_GRU"
# WEIGHTS_NAME="osd_1_45m_mae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type GRU \
#             RL.POLICY.randomize_augmentations_over_envs False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="mae_finetuned_LSTM"
# WEIGHTS_NAME="osd_1_45m_mae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False"
# SPLIT="train_extra"
# NUM_ENV=10
# run_training 0

# EXP_NAME="mae_small_finetuned_LSTM"
# WEIGHTS_NAME="mae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="tmae_finetuned_LSTM_randomised_envs"
# WEIGHTS_NAME="tmae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs True"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_finetuned_masking_0_75"
# WEIGHTS_NAME="tmae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs True \
#             RL.POLICY.vit_mask_ratio 0.75"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="tmae_finetuned_masking_0_25"
# WEIGHTS_NAME="tmae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs True \
#             RL.POLICY.vit_mask_ratio 0.25"
# SPLIT="train_extra"
# NUM_ENV=5
# NODES=8
# run_training 0

# EXP_NAME="tmae_finetuned_masking_0"
# WEIGHTS_NAME="tmae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs True \
#             RL.POLICY.vit_mask_ratio 0.0"
# SPLIT="train_extra"
# NUM_ENV=5
# NODES=8
# run_training 0

EXP_NAME="tmae_base_finetuned_LSTM_compression_layer"
WEIGHTS_NAME="tmae_base_01.pth"
BACKBONE="vit_base_patch16"
EXTRA_CMDS="RL.POLICY.freeze_backbone False \
            RL.POLICY.rnn_type LSTM \
            RL.PPO.lr 2.5e-4"
SPLIT="train_extra"
run_training 0

# EXP_NAME="tmae_small_finetuned_LSTM"
# WEIGHTS_NAME="tmae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="tmae_small_finetuned_LSTM_no_augs"
# WEIGHTS_NAME="tmae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.use_augmentations_test_time False \
#             RL.POLICY.use_augmentations False \
#             RL.PPO.lr 2.5e-4"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="tmae_small_finetuned_LSTM_no_cj"
# WEIGHTS_NAME="tmae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4 \
#             RL.POLICY.augmentations_name shift"
# SPLIT="train_extra"
# run_training 0

EXP_NAME="tmae_small_finetuned_LSTM_randomized_envs"
WEIGHTS_NAME="tmae_small_01.pth"
BACKBONE="vit_small_patch16"
EXTRA_CMDS="RL.POLICY.freeze_backbone False \
            RL.POLICY.rnn_type LSTM \
            RL.PPO.lr 2.5e-4 \
            RL.POLICY.randomize_augmentations_over_envs True"
SPLIT="train_extra"
run_training 0


# EXP_NAME="mae_scratch_first_experiment"
# WEIGHTS_NAME=""
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False"
# SPLIT="train_extra"
# NUM_ENV=6
# run_training 0
