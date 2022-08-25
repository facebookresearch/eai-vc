#!/bin/bash
#SBATCH --job-name mae
#SBATCH --output log.out
#SBATCH --error log.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 56
#SBATCH --ntasks-per-node 1
#SBATCH --partition long
#SBATCH --constraint a40

cd mae
source activate eai

DATA="data/datasets/hm3d+gibson/v1/train"

set -x
srun torchrun \
--nproc_per_node 8 \
main_pretrain.py \
--batch_size 512 \
--epochs 800 \
--accum_iter 1 \
--model mae_vit_small_patch16 \
--mask_ratio 0.75 \
--norm_pix_loss \
--weight_decay 0.05 \
--blr 1.5e-4 \
--warmup_epochs 40 \
--data_path $DATA \
--seed $RANDOM \
--num_workers 5 \
--wandb_name "mae-$SLURM_JOB_ID" \
--wandb_mode "online" \