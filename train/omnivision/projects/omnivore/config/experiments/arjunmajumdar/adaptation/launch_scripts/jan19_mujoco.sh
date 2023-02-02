#!/bin/bash

# config
ADAPT_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/005_mae_adapt_v1.yaml"

# checkpoint
STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_large_ego_imagenet_inav_182_epochs.pth"

# =================================
#

ADROIT_MANIFEST="/checkpoint/maksymets/eaif/datasets/adroit-expert-v0.1/manifest.txt"

# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${ADROIT_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=24

#
# =================================

# =================================
#

METAWORLD_MANIFEST="/checkpoint/maksymets/eaif/datasets/metaworld-expert-v0.1/manifest.txt"

# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${METAWORLD_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=24

#
# =================================

# =================================
#

DMC_MANIFEST="/checkpoint/maksymets/eaif/datasets/dmc-expert-v0.1/manifest.txt"

# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${DMC_MANIFEST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}] \
submitit.timeout_hour=24

#
# =================================
