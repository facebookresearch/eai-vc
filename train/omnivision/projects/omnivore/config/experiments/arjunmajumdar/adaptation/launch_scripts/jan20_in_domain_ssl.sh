#!/bin/bash

# config
SCRATCH_MAE_CONFIG_FILE="config/experiments/arjunmajumdar/adaptation/006_mae_scratch_v1.yaml"


# =================================
# 1

GIBSON_MANIFEST="/checkpoint/maksymets/eaif/datasets/manifests/gibson_manifest.txt"

./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${GIBSON_MANIFEST}] \
submitit.partition="'devlab,learnlab'" \
submitit.timeout_hour=24

#
# =================================

# =================================
# 2

REARRANGE_MANIFEST="/checkpoint/maksymets/eaif/datasets/rearrange/manifest.txt"

./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${REARRANGE_MANIFEST}] \
submitit.partition="'devlab,learnlab'" \
submitit.timeout_hour=60

#
# =================================

# =================================
# 3

OBJECTNAV_MANIFEST="/checkpoint/maksymets/eaif/datasets/manifests/objectnav_demos_100k_manifest.txt"

./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${OBJECTNAV_MANIFEST}] \
submitit.partition="'devlab,learnlab'" \
submitit.timeout_hour=24

#
# =================================

# =================================
# 4

MUJOCO_MANIFEST="/checkpoint/maksymets/eaif/datasets/mujoco-combined-expert-v0.1/manifest.txt"

./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${MUJOCO_MANIFEST}] \
submitit.partition="'devlab,learnlab'" \
submitit.timeout_hour=60

#
# =================================

# =================================
# 5

TRI_FINGER_MANIFEST="/checkpoint/maksymets/eaif/datasets/manifests/trifinger_90k_manifest.txt"

./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${TRI_FINGER_MANIFEST}] \
submitit.partition="'devlab,learnlab'" \
submitit.timeout_hour=24

#
# =================================

# =================================
# 6

ADROIT_MANIFEST="/checkpoint/maksymets/eaif/datasets/adroit-expert-v0.1/manifest.txt"

./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${ADROIT_MANIFEST}] \
submitit.partition="'devlab,learnlab'" \
submitit.timeout_hour=24

#
# =================================

# =================================
# 7

METAWORLD_MANIFEST="/checkpoint/maksymets/eaif/datasets/metaworld-expert-v0.1/manifest.txt"

./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${METAWORLD_MANIFEST}] \
submitit.partition="'devlab,learnlab'" \
submitit.timeout_hour=24

#
# =================================

# =================================
# 8

DMC_MANIFEST="/checkpoint/maksymets/eaif/datasets/dmc-expert-v0.1/manifest.txt"

./dev/launch_job.py \
-c ${SCRATCH_MAE_CONFIG_FILE} \
--extra-hydra-overrides \
trainer.data.train.dataset.path_file_list=[${DMC_MANIFEST}] \
submitit.partition="'devlab,learnlab'" \
submitit.timeout_hour=36

#
# =================================
