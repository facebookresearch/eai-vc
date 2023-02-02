#!/bin/bash

# config
ADAPT_TMAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/004_tmae_adapt_v1.yaml"
ADAPT_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/005_mae_adapt_v1.yaml"

# checkpoint
STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_large_ego_imagenet_inav_182_epochs.pth"

# =================================
# 3 - Gibson MAE

GIBSON_MANIFEST="/checkpoint/yixinlin/eaif/datasets/manifests/gibson_manifest.txt"

# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${GIBSON_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=24

#
# =================================

# =================================
# 3 - Gibson TMAE

REPEAT_FACTOR=51
GIBSON_MANIFEST="/checkpoint/yixinlin/eaif/datasets/manifests/gibson_manifest.txt"

# TMAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${GIBSON_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=36

#
# =================================
