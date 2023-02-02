#!/bin/bash

# set conda environment
conda activate /private/home/yixinlin/miniconda3/envs/ov

ADAPT_TMAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/004_tmae_adapt_vitl_v0.yaml"

STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_large_ego_imagenet_inav_182_epochs.pth"

# =================================
# MuJoCo (Adroit + DMControl + MetaWorld)

REPEAT_FACTOR=415
MUJOCO_MANIFEST="/checkpoint/yixinlin/eaif/datasets/mujoco-combined-expert-v0.1/manifest.txt"

# TMAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${MUJOCO_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================

# =================================
# TriFinger

REPEAT_FACTOR=151
TRI_FINGER_MANIFEST="/checkpoint/yixinlin/eaif/datasets/manifests/trifinger_90k_manifest.txt"

# TMAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${TRI_FINGER_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================

# =================================
# Gibson

REPEAT_FACTOR=51
GIBSON_MANIFEST="/checkpoint/yixinlin/eaif/datasets/manifests/gibson_manifest.txt"

# TMAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${GIBSON_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================

# =================================
# Rearrange

REPEAT_FACTOR=21
REARRANGE_MANIFEST="/checkpoint/yixinlin/eaif/datasets/rearrange/manifest.txt"

# TMAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${REARRANGE_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================
