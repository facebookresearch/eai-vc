#!/bin/bash

# config
ADAPT_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/005_mae_adapt_v1.yaml"

# checkpoint
STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_large_ego_imagenet_inav_182_epochs.pth"

# =================================
# ObjectNav -- MAE

OBJECTNAV_MANIFEST="/checkpoint/maksymets/eaif/datasets/manifests/objectnav_demos_100k_manifest.txt"

# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${OBJECTNAV_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=36

#
# =================================
