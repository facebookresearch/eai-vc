#!/bin/bash

CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/004_tmae_adapt_v1.yaml"

# Gibson
REPEAT_FACTOR=51
PATH_FILE_LIST="/checkpoint/yixinlin/eaif/datasets/manifests/gibson_manifest.txt"
CHECKPOINT_PATHS="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_large_ego_inav_233_epochs.pth"

# # GT paths
# PATH_FILE_LIST="/srv/flash1/amajumdar36/image-datasets/gibson-manifest.txt"
# CHECKPOINT_PATHS="/srv/flash1/amajumdar36/eaif-models/mae_vit_large_ego_inav_233_epochs.pth"

./dev/launch_job.py --local \
-c ${CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${PATH_FILE_LIST}] \
trainer.model.state_dict.checkpoint_paths=[${CHECKPOINT_PATHS}] \
