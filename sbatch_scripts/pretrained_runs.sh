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
RUN_TRAIN_SCRIPT=false
RUN_EVAL_SCRIPT=true

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


# EXP_NAME="tmae_base_finetuned_LSTM_global_pool"
# WEIGHTS_NAME="tmae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_base_finetuned_LSTM_global_pool_masking_v2"
# WEIGHTS_NAME="tmae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True \
#             RL.POLICY.vit_use_cls False \
#             RL.POLICY.randomize_augmentations_over_envs True"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_base_finetuned_LSTM_global_pool_masking_v2_no_random_aug"
# WEIGHTS_NAME="tmae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True \
#             RL.POLICY.vit_use_cls False \
#             RL.POLICY.randomize_augmentations_over_envs False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_base_finetuned_LSTM_compression_layer"
# WEIGHTS_NAME="tmae_base_01.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4"
# SPLIT="train_extra"
# NUM_ENV=5
# NODES=8
# run_training 0

# EXP_NAME="tmae_small_finetuned_LSTM"
# WEIGHTS_NAME="tmae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="tmae_small_finetuned_LSTM_global_pool"
# WEIGHTS_NAME="tmae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="tmae_small_finetuned_LSTM_global_pool_fc_norm"
# WEIGHTS_NAME="tmae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="tmae_small_finetuned_LSTM_cls_token"
# WEIGHTS_NAME="tmae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4 \
#             RL.POLICY.vit_use_cls True \
#             RL.POLICY.vit_mask_ratio 0.5"
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

# EXP_NAME="tmae_small_finetuned_LSTM_randomized_envs"
# WEIGHTS_NAME="tmae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4 \
#             RL.POLICY.randomize_augmentations_over_envs True"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="tmae_small_finetuned_LSTM_cj_pretrained"
# WEIGHTS_NAME="tmae_small_01_cj.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.PPO.lr 2.5e-4 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5"
# SPLIT="train_extra"
# run_training 0

# EXP_NAME="mae_scratch_first_experiment"
# WEIGHTS_NAME=""
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False"
# SPLIT="train_extra"
# NUM_ENV=6
# run_training 0

# EXP_NAME="mae_base_399_finetuned_new_data_nozer"
# WEIGHTS_NAME="mae_base_02_399_ckpt.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="mae_base_799_finetuned_new_data"
# WEIGHTS_NAME="mae_base_01_799_ckpt.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="mae_small_finetuned_new_data"
# WEIGHTS_NAME="mae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="tmae_improved_embedding"
# WEIGHTS_NAME="tmae_small_02.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="mae_improved_cj"
# WEIGHTS_NAME="mae_base_01_cj.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_1_100"
# WEIGHTS_NAME="tmae_small_offset_1_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="tmae_small_offset_1_400"
# WEIGHTS_NAME="tmae_small_offset_1_399.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="tmae_improved_embedding_frozen"
# WEIGHTS_NAME="tmae_small_02.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="tmae_improved_embedding_compression_layer"
# WEIGHTS_NAME="tmae_small_02.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_improved_embedding_frozen_no_aug_new_reward"
# WEIGHTS_NAME="tmae_small_02.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.POLICY.use_augmentations False \
#             RL.POLICY.use_augmentations_test_time False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# TEST_EPISODE_COUNT=30
# VIDEO_OPTION="[\"wandb\"]"
# WANDB_MODE="offline"
# CHKP_NAME="ckpt.99.pth"
# run_training 0

# EXP_NAME="mae_improved_embedding_frozen"
# WEIGHTS_NAME="mae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="tmae_small_offset_1_100_frozen"
# WEIGHTS_NAME="tmae_small_offset_1_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="tmae_small_offset_1_400_frozen"
# WEIGHTS_NAME="tmae_small_offset_1_399.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="mae_base_improved_cj_frozen"
# WEIGHTS_NAME="mae_base_01_cj.pth"
# BACKBONE="vit_base_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="tmae_small_offset_1_100_frozen_new_reward"
# WEIGHTS_NAME="tmae_small_offset_1_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone True \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="tmae_small_offset_1_10_new_reward"
# WEIGHTS_NAME="tmae_small_offset_1_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool True \
#             RL.POLICY.vit_mask_ratio 0.5 \
#             RL.POLICY.vit_use_fc_norm True"
# SPLIT="train_extra"
# NUM_ENV=10
# NODES=4
# run_training 0

# EXP_NAME="tmae_small_offset_1_10_new_reward_compression_layer"
# WEIGHTS_NAME="tmae_small_offset_1_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_1_100_compression_layer_adamw_1e-6"
# WEIGHTS_NAME="tmae_small_offset_1_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_1_100_compression_layer_adamw_1e-5"
# WEIGHTS_NAME="tmae_small_offset_1_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.PPO.wd 1e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_1_100_compression_layer_adamw_1e-4"
# WEIGHTS_NAME="tmae_small_offset_1_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.PPO.wd 1e-4 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_1_100_compression_layer_adamw_1e-3"
# WEIGHTS_NAME="tmae_small_offset_1_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.PPO.wd 1e-3 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0


# EXP_NAME="tmae_small_offset_1_100_compression_layer_adamw_0"
# WEIGHTS_NAME="tmae_small_offset_1_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.PPO.wd 0.0 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_100_random_views_reward_hack"
# WEIGHTS_NAME="tmae_small_offset_4_random_views_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="mae_improved_embedding_better_reward"
# WEIGHTS_NAME="mae_small_01.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="mae_small_01_100_better_reward"
# WEIGHTS_NAME="mae_small_01_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_100_random_views_droppath_0.05"
# WEIGHTS_NAME="tmae_small_offset_4_random_views_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False \
#             RL.POLICY.drop_path_rate 0.05"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_100_random_views_droppath_0.1"
# WEIGHTS_NAME="tmae_small_offset_4_random_views_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False \
#             RL.POLICY.drop_path_rate 0.1"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_100_random_views_droppath_0.2"
# WEIGHTS_NAME="tmae_small_offset_4_random_views_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False \
#             RL.POLICY.drop_path_rate 0.2"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_100_random_views_droppath_0.3"
# WEIGHTS_NAME="tmae_small_offset_4_random_views_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False \
#             RL.POLICY.drop_path_rate 0.3"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_100_tiny_decoder"
# WEIGHTS_NAME="tmae_small_offset_4_tiny_decoder.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_140_tiny_decoder"
# WEIGHTS_NAME="tmae_small_offset_4_tiny_decoder_140.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_200_tiny_decoder"
# WEIGHTS_NAME="tmae_small_offset_4_tiny_decoder_200.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_399_tiny_decoder"
# WEIGHTS_NAME="tmae_small_offset_4_tiny_decoder_399.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_100_large_decoder"
# WEIGHTS_NAME="tmae_small_offset_4_large_decoder_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_100_random_views_reward_hack_0.8"
# WEIGHTS_NAME="tmae_small_offset_4_random_views_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False \
#             TASK_CONFIG.TASK.TRAIN_SUCCESS.SUCCESS_DISTANCE 0.8"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

# EXP_NAME="tmae_small_offset_4_100_random_views_reward_hack_0.7"
# WEIGHTS_NAME="tmae_small_offset_4_random_views_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False \
#             TASK_CONFIG.TASK.TRAIN_SUCCESS.SUCCESS_DISTANCE 0.7"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0

EXP_NAME="tmae_small_offset_4_real_estate_100"
WEIGHTS_NAME="tmae_small_offset_4_real_estate_100.pth"
BACKBONE="vit_small_patch16"
EXTRA_CMDS="RL.POLICY.freeze_backbone False \
            RL.POLICY.rnn_type LSTM \
            RL.POLICY.randomize_augmentations_over_envs False \
            RL.PPO.lr 6.25e-5 \
            RL.POLICY.vit_global_pool False \
            RL.POLICY.vit_use_fc_norm False"
SPLIT="train_extra"
NUM_ENV=8
NODES=5
run_training 0

# EXP_NAME="tmae_small_offset_4_real_estate_hm3d_gibson_100"
# WEIGHTS_NAME="tmae_small_offset_4_real_estate_hm3d_gibson_100.pth"
# BACKBONE="vit_small_patch16"
# EXTRA_CMDS="RL.POLICY.freeze_backbone False \
#             RL.POLICY.rnn_type LSTM \
#             RL.POLICY.randomize_augmentations_over_envs False \
#             RL.PPO.lr 6.25e-5 \
#             RL.POLICY.vit_global_pool False \
#             RL.POLICY.vit_use_fc_norm False"
# SPLIT="train_extra"
# NUM_ENV=8
# NODES=5
# run_training 0