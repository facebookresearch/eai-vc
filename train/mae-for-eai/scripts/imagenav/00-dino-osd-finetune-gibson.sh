#!/bin/bash
#SBATCH --job-name mae
#SBATCH --output log.out
#SBATCH --error log.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 7
#SBATCH --ntasks-per-node 8
#SBATCH --signal USR1@600
#SBATCH --partition short
#SBATCH --constraint a40

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
RL.POLICY.backbone resnet50 \
RL.POLICY.pretrained_encoder data/models/omnidata_DINO_02.pth \
RL.POLICY.freeze_backbone False \
