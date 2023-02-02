#!/bin/bash

# wandb login --host=https://api.wandb.ai --relogin

SCRATCH_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/002_mae_scratch_v0.yaml"
ADAPT_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/003_mae_adapt_v0.yaml"

# STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_inav_233_epochs.pth"
STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_imagenet_inav_182_epochs.pth"

# =================================
# Adroit

ADROIT_MANIFEST="/checkpoint/yixinlin/eaif/datasets/adroit-expert-v0.1/manifest.txt"

# MAE from scratch
./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${ADROIT_MANIFEST}]


# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${ADROIT_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================

# =================================
# DMControl

DMCONTROL_MANIFEST="/checkpoint/yixinlin/eaif/datasets/dmc-expert-v0.1/manifest.txt"

# MAE from scratch
./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${DMCONTROL_MANIFEST}]


# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${DMCONTROL_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================

# =================================
# MetaWorld

META_WORLD_MANIFEST="/checkpoint/yixinlin/eaif/datasets/metaworld-expert-v0.1/manifest.txt"

# MAE from scratch
./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${META_WORLD_MANIFEST}]


# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${META_WORLD_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================

# =================================
# MuJoCo

MUJOCO_MANIFEST="/checkpoint/yixinlin/eaif/datasets/mujoco-combined-expert-v0.1/manifest.txt"

# MAE from scratch
./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${MUJOCO_MANIFEST}]


# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${MUJOCO_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================

# =================================
# Gibson

GIBSON_MANIFEST="/checkpoint/yixinlin/eaif/datasets/manifests/gibson_manifest.txt"

# MAE from scratch
./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${GIBSON_MANIFEST}]


# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${GIBSON_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================

# =================================
# TriFinger

TRI_FINGER_MANIFEST="/checkpoint/yixinlin/eaif/datasets/manifests/trifinger_90k_manifest.txt"

# MAE from scratch
./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${TRI_FINGER_MANIFEST}]


# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${TRI_FINGER_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================

# =================================
# Rearrange

REARRANGE_MANIFEST="/checkpoint/yixinlin/eaif/datasets/rearrange/manifest.txt"

# MAE from scratch
./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${REARRANGE_MANIFEST}]


# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${REARRANGE_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================

# # =================================
# # RoboPen

# ROBO_PEN_MANIFEST=??

# # MAE from scratch
# ./dev/launch_job.py \
# -c ${SCRATCH_MAE_CONFIG_FILE} \
# --extra-hydra-overrides \
# trainer.data.train.dataset.path_file_list=[${ROBO_PEN_MANIFEST}]


# # MAE from pretrained checkpoint
# ./dev/launch_job.py \
# -c ${ADAPT_MAE_CONFIG_FILE} \
# --extra-hydra-overrides \
# trainer.data.train.dataset.path_file_list=[${ROBO_PEN_MANIFEST}] \
# trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

# #
# # =================================
