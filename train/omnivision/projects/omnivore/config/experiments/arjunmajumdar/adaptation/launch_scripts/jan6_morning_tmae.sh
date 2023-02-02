#!/bin/bash

# Notes:
# - set `REPEAT_FACTOR` to "number of files" / "number of folders"

# wandb login --host=https://api.wandb.ai --relogin

# set conda environment
conda activate /private/home/yixinlin/miniconda3/envs/ov

SCRATCH_TMAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/000_tmae_scratch_v0.yaml"
ADAPT_TMAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/001_tmae_adapt_v0.yaml"

# STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_inav_233_epochs.pth"
STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_imagenet_inav_182_epochs.pth"

# =================================
# MuJoCo (Adroit + DMControl + MetaWorld)

REPEAT_FACTOR=415
MUJOCO_MANIFEST="/checkpoint/yixinlin/eaif/datasets/mujoco-combined-expert-v0.1/manifest.txt"

# # TMAE from scratch
# ./dev/launch_job.py \
# -c ${SCRATCH_TMAE_CONFIG_FILE} \
# --extra-hydra-overrides \
# trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
# trainer.data.train.dataset.path_file_list=[${MUJOCO_MANIFEST}] \
# submitit.timeout_hour=60


# TMAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${MUJOCO_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=60

#
# =================================
