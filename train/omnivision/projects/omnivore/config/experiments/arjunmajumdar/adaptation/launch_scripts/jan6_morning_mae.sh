#!/bin/bash

# wandb login --host=https://api.wandb.ai --relogin

# set conda environment
conda activate /private/home/yixinlin/miniconda3/envs/ov

SCRATCH_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/002_mae_scratch_v0.yaml"
ADAPT_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/003_mae_adapt_v0.yaml"

# STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_inav_233_epochs.pth"
STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_imagenet_inav_182_epochs.pth"

# =================================
# MuJoCo

MUJOCO_MANIFEST="/checkpoint/yixinlin/eaif/datasets/mujoco-combined-expert-v0.1/manifest.txt"

# # MAE from scratch
# ./dev/launch_job.py \
# -c ${SCRATCH_MAE_CONFIG_FILE} \
# --extra-hydra-overrides \
# trainer.data.train.dataset.path_file_list=[${MUJOCO_MANIFEST}]


# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${MUJOCO_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=36

#
# =================================
