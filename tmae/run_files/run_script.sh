python submitit_pretrain.py \
    --wandb_name tmae_vit_base_01 \
    --nodes 4 \
    --batch_size 64 \
    --accum_iter 2 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --max_offset 16 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name tmae_vit_small_01 \
    --nodes 4 \
    --batch_size 128 \
    --accum_iter 1 \
    --model mae_vit_small_patch16 \
    --norm_pix_loss \
    --max_offset 16 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name tmae_vit_base_lr_01 \
    --nodes 4 \
    --batch_size 64 \
    --accum_iter 2 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --max_offset 16 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 5.0e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name tmae_vit_base_next_image_only_01 \
    --nodes 4 \
    --batch_size 64 \
    --accum_iter 2 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --max_offset 16 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 1.0 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name tmae_vit_small_01_cj \
    --nodes 4 \
    --batch_size 128 \
    --accum_iter 1 \
    --model mae_vit_small_patch16 \
    --norm_pix_loss \
    --max_offset 16 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32 --color_jitter

python submitit_pretrain.py \
    --wandb_name tmae_vit_base_01_offset_4 \
    --nodes 4 \
    --batch_size 64 \
    --accum_iter 2 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --max_offset 4 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name tmae_vit_base_01_offset_8 \
    --nodes 4 \
    --batch_size 64 \
    --accum_iter 2 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --max_offset 8 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name tmae_vit_base_01_ratio_75_75 \
    --nodes 4 \
    --batch_size 64 \
    --accum_iter 2 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --max_offset 16 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.75 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32