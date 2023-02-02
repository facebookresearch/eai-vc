#!/bin/bash

# config
ADAPT_TMAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/001_tmae_adapt_v0.yaml"

# checkpoint
STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_imagenet_inav_182_epochs.pth"

# =================================
# Rearrange

REPEAT_FACTOR=21
REARRANGE_MANIFEST="/checkpoint/yixinlin/eaif/datasets/rearrange/manifest.txt"

# 400 epochs
./dev/launch_job.py \
-c ${ADAPT_TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.max_epochs=400 \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${REARRANGE_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=6

# 200 epochs
./dev/launch_job.py \
-c ${ADAPT_TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.max_epochs=200 \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${REARRANGE_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=6

#
# =================================
