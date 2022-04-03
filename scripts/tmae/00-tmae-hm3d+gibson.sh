#!/bin/bash
#SBATCH --job-name tmae
#SBATCH --output log.out
#SBATCH --error log.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 56
#SBATCH --ntasks-per-node 1
#SBATCH --partition short
#SBATCH --constraint a40

cd tmae
source activate eai

DATA="data/datasets/hm3d+gibson/v1/train"

set -x
srun torchrun \
--nproc_per_node 8 \
main_pretrain.py \
--batch_size 128 \
--epochs 800 \
--accum_iter 4 \
--model mae_vit_base_patch16 \
--max_offset 16 \
--mask_ratio1 0.75 \
--mask_ratio2 0.95 \
--loss_weight 0.5 \
--norm_pix_loss \
--weight_decay 0.05 \
--blr 1.5e-4 \
--warmup_epochs 40 \
--data_path $DATA \
--seed $RANDOM \
--num_workers 5 \
--wandb_name "tmae-$SLURM_JOB_ID" \
--wandb_mode "online" \
