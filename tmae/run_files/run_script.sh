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

python submitit_pretrain.py \
    --wandb_name tmae_small_improved_embed \
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
    --wandb_name tmae_small_offset_1 \
    --nodes 4 \
    --batch_size 128 \
    --accum_iter 1 \
    --model mae_vit_small_patch16 \
    --norm_pix_loss \
    --max_offset 1 \
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
    --wandb_name tmae_small_offset_4_viz \
    --nodes 4 \
    --batch_size 128 \
    --accum_iter 1 \
    --model mae_vit_small_patch16 \
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
    --wandb_name tmae_small_offset_4_tiny_decoder \
    --nodes 4 \
    --batch_size 128 \
    --accum_iter 1 \
    --model mae_vit_small_patch16_tiny_decoder \
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
    --wandb_name tmae_small_offset_4_large_decoder \
    --nodes 8 \
    --batch_size 64 \
    --accum_iter 1 \
    --model mae_vit_small_patch16_large_decoder \
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
    --wandb_name tmae_small_offset_4_real_estate \
    --nodes 4 \
    --batch_size 128 \
    --accum_iter 1 \
    --model mae_vit_small_patch16 \
    --norm_pix_loss \
    --max_offset 4 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/real-estate-10k-frames-v0/ \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name tmae_small_offset_4_real_estate_hm3d_gibson \
    --nodes 4 \
    --batch_size 128 \
    --accum_iter 1 \
    --model mae_vit_small_patch16 \
    --norm_pix_loss \
    --max_offset 4 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/real-estate-10k-frames-v0/ /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name tmae_small_offset_4 \
    --nodes 4 \
    --batch_size 128 \
    --accum_iter 1 \
    --model mae_vit_small_patch16 \
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
    --wandb_name tmae_small_decoder_large_offset_4_RE10k_HGSP_random_views \
    --nodes 8 \
    --batch_size 64 \
    --accum_iter 1 \
    --model mae_vit_small_patch16_large_decoder \
    --norm_pix_loss \
    --max_offset 4 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --randomize_views \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/real-estate-10k-frames-v0/ /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name tmae_small_decoder_large_offset_4_RE10k_HGSP_ego4d_random_views \
    --nodes 8 \
    --batch_size 64 \
    --accum_iter 1 \
    --model mae_vit_small_patch16_large_decoder \
    --norm_pix_loss \
    --max_offset 4 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --randomize_views \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/real-estate-10k-frames-v0/ /checkpoint/karmeshyadav/hm3d+gibson/v1/train/ /checkpoint/karmeshyadav/ego4d_images_v0/ \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name tmae_small_decoder_large_offset_4_RE10k_HGSP_random_views_95_95_masking \
    --nodes 8 \
    --batch_size 64 \
    --accum_iter 1 \
    --model mae_vit_small_patch16_large_decoder \
    --norm_pix_loss \
    --max_offset 4 \
    --mask_ratio1 0.95 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --randomize_views \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/real-estate-10k-frames-v0/ /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name tmae_small_decoder_large_offset_16_RE10k_HGSP_random_views \
    --nodes 8 \
    --batch_size 64 \
    --accum_iter 1 \
    --model mae_vit_small_patch16_large_decoder \
    --norm_pix_loss \
    --max_offset 16 \
    --mask_ratio1 0.75 \
    --mask_ratio2 0.95 \
    --loss_weight 0.5 \
    --epochs 400 \
    --warmup_epochs 40 \
    --randomize_views \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/real-estate-10k-frames-v0/ /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32