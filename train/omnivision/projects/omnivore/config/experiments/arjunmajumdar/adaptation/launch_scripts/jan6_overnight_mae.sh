#!/bin/bash

ADAPT_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/005_mae_adapt_vitl_v0.yaml"

STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_large_ego_imagenet_inav_182_epochs.pth"

# =================================
# MuJoCo (Adroit + DMControl + MetaWorld)

MUJOCO_MANIFEST="/checkpoint/yixinlin/eaif/datasets/mujoco-combined-expert-v0.1/manifest.txt"

# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${MUJOCO_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=60

#
# =================================

# =================================
# TriFinger

TRI_FINGER_MANIFEST="/checkpoint/yixinlin/eaif/datasets/manifests/trifinger_90k_manifest.txt"

# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${TRI_FINGER_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=36

#
# =================================

# =================================
# Gibson

GIBSON_MANIFEST="/checkpoint/yixinlin/eaif/datasets/manifests/gibson_manifest.txt"

# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${GIBSON_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=36

#
# =================================

# =================================
# Rearrange

REARRANGE_MANIFEST="/checkpoint/yixinlin/eaif/datasets/rearrange/manifest.txt"

# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${REARRANGE_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=18

#
# =================================
