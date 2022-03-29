#!/bin/bash
#SBATCH --job-name mae
#SBATCH --output log.out
#SBATCH --error log.err
#SBATCH --gres gpu:8
#SBATCH --nodes 2
#SBATCH --cpus-per-task 7
#SBATCH --ntasks-per-node 8
#SBATCH --partition short
#SBATCH --constraint a40

DATA="data/datasets/hm3d+gibson/v1/train"

source activate eai

set -x
srun python -u \
main_pretrain.py \
--batch_size 128 \
--epochs 800 \
--accum_iter 4 \
--model mae_vit_base_patch16 \
--mask_ratio 0.75 \
--norm_pix_loss \
--weight_decay 0.05 \
--blr 1.5e-4 \
--warmup_epochs 40 \
--data_path $DATA \
--seed $RANDOM \
--num_workers 5 \
--wandb_name "mae-%j" \
--wandb_mode "online" \
