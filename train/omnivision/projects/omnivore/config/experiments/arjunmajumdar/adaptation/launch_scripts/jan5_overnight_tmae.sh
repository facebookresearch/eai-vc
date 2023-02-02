#!/bin/bash

# Notes:
# - set `REPEAT_FACTOR` to "number of files" / "number of folders"

# wandb login --host=https://api.wandb.ai --relogin

SCRATCH_TMAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/000_tmae_scratch_v0.yaml"
ADAPT_TMAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/001_tmae_adapt_v0.yaml"

STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_imagenet_inav_182_epochs.pth"

# =================================
# Adroit

REPEAT_FACTOR=150
ADROIT_MANIFEST="/checkpoint/yixinlin/eaif/datasets/adroit-expert-v0.1/manifest.txt"

# # TMAE from scratch
# ./dev/launch_job.py \
# -c ${SCRATCH_TMAE_CONFIG_FILE} \
# --extra-hydra-overrides \
# trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
# trainer.data.train.dataset.path_file_list=[${ADROIT_MANIFEST}]


# TMAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${ADROIT_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================

# =================================
# Gibson

REPEAT_FACTOR=51
GIBSON_MANIFEST="/checkpoint/yixinlin/eaif/datasets/manifests/gibson_manifest.txt"

# # TMAE from scratch
# ./dev/launch_job.py \
# -c ${SCRATCH_TMAE_CONFIG_FILE} \
# --extra-hydra-overrides \
# trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
# trainer.data.train.dataset.path_file_list=[${GIBSON_MANIFEST}]


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

# # TMAE from scratch
# ./dev/launch_job.py \
# -c ${SCRATCH_TMAE_CONFIG_FILE} \
# --extra-hydra-overrides \
# trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
# trainer.data.train.dataset.path_file_list=[${REARRANGE_MANIFEST}]


# TMAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${REARRANGE_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

#
# =================================
