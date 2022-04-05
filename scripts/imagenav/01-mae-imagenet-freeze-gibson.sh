#!/bin/bash
#SBATCH --job-name imagenav
#SBATCH --output log.out
#SBATCH --error log.err
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 7
#SBATCH --ntasks-per-node 4
#SBATCH --signal USR1@600
#SBATCH --partition short
#SBATCH --constraint a40
#SBATCH --requeue

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

source activate eai

set -x
srun \
python -u run.py \
--exp-config configs/experiments/imagenav.yaml \
--run-type train \
NUM_ENVIRONMENTS 20 \
RL.POLICY.backbone vit_base_patch16 \
RL.POLICY.vit_use_fc_norm False \
RL.POLICY.vit_global_pool False \
RL.POLICY.vit_mask_ratio None \
RL.POLICY.pretrained_encoder data/models/osd_1_45m_mae_base_01.pth \
RL.POLICY.freeze_backbone True \
