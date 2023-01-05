#!/bin/bash

# Notes:
# - set `REPEAT_FACTOR` to "number of files" / "number of folders"

TMAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/001_tmae_adapt_v0.yaml"
MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/003_mae_adapt_v0.yaml"

CHECKPOINT_PATHS="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_inav_233_epochs.pth"

# -----------------
# gibson
# -----------------
REPEAT_FACTOR=51
PATH_FILE_LIST="/checkpoint/yixinlin/eaif/datasets/manifests/gibson_manifest.txt"

# tmae
./dev/launch_job.py \
-c ${TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${PATH_FILE_LIST}] \
trainer.model.state_dict.checkpoint_paths=[${CHECKPOINT_PATHS}]

# mae
./dev/launch_job.py \
-c ${MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${PATH_FILE_LIST}] \
trainer.model.state_dict.checkpoint_paths=[${CHECKPOINT_PATHS}]

# -----------------
# adroit
# -----------------

REPEAT_FACTOR=???
PATH_FILE_LIST=???

# tmae
./dev/launch_job.py \
-c ${TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${PATH_FILE_LIST}] \
trainer.model.state_dict.checkpoint_paths=[${CHECKPOINT_PATHS}]

# mae
./dev/launch_job.py \
-c ${MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${PATH_FILE_LIST}] \
trainer.model.state_dict.checkpoint_paths=[${CHECKPOINT_PATHS}]
