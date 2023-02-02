#!/bin/bash

# config
SCRATCH_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/006_mae_scratch_v1.yaml"


# =================================
#

ADROIT_MANIFEST="/checkpoint/maksymets/eaif/datasets/adroit-expert-v0.1/manifest.txt"

./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.optim.gradient_clip.max_norm=0.01 \
trainer.data.train.dataset.path_file_list=[${ADROIT_MANIFEST}] \
submitit.partition="learnlab" \
submitit.timeout_hour=36

#
# =================================


# =================================
#

ADROIT_MANIFEST="/checkpoint/maksymets/eaif/datasets/adroit-expert-v0.1/manifest.txt"

./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.optim.gradient_clip.max_norm=0.002 \
trainer.data.train.dataset.path_file_list=[${ADROIT_MANIFEST}] \
submitit.partition="learnlab" \
submitit.timeout_hour=36

#
# =================================

# =================================
#

ADROIT_MANIFEST="/checkpoint/maksymets/eaif/datasets/adroit-expert-v0.1/manifest.txt"

./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.optim.gradient_clip.max_norm=0.0002 \
trainer.data.train.dataset.path_file_list=[${ADROIT_MANIFEST}] \
submitit.partition="learnlab" \
submitit.timeout_hour=36

#
# =================================
