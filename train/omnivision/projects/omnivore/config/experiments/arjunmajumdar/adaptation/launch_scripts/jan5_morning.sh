#!/bin/bash

# Notes:
# - set `REPEAT_FACTOR` to "number of files" / "number of folders"

wandb login --host=https://api.wandb.ai --relogin

SCRATCH_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/002_mae_scratch_v0.yaml"
SCRATCH_TMAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/000_tmae_scratch_v0.yaml"

ADAPT_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/003_mae_adapt_v0.yaml"
ADAPT_TMAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/001_tmae_adapt_v0.yaml"

STARTING_POINT="/checkpoint/yixinlin/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_inav_233_epochs.pth"

# =================================
# Jan 5th morning runs
# =================================

# Gibson
REPEAT_FACTOR=51
PATH_FILE_LIST="/checkpoint/yixinlin/eaif/datasets/manifests/gibson_manifest.txt"

# TMAE from scratch
./dev/launch_job.py \
-c ${SCRATCH_TMAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.repeat_factor=${REPEAT_FACTOR} \
trainer.data.train.dataset.path_file_list=[${PATH_FILE_LIST}]

# MAE from pretrained checkpoint
./dev/launch_job.py \
-c ${ADAPT_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${PATH_FILE_LIST}] \
trainer.model.state_dict.checkpoint_paths=[${STARTING_POINT}]

# MAE from scratch
./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${PATH_FILE_LIST}]
