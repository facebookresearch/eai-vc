# python submitit_pretrain.py \
#     --wandb_name imagenet_mae_reproduce \
#     --nodes 8 \
#     --batch_size 64 \
#     --model mae_vit_large_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 800 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path /checkpoint/karmeshyadav/imagenet_full_size/ \
#     --dataset_type imagenet 

# python submitit_pretrain.py \
#     --wandb_name imagenet_mae_reproduce_01 \
#     --nodes 8 \
#     --batch_size 64 \
#     --model mae_vit_large_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 800 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path /checkpoint/karmeshyadav/imagenet_full_size/ \
#     --dataset_type imagenet --partition learnlab

# python submitit_pretrain.py \
#     --wandb_name osd_1_45_default_params_01 \
#     --nodes 8 \
#     --batch_size 64 \
#     --model mae_vit_large_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 800 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path /checkpoint/karmeshyadav/omnidataset/ \
#     --dataset_type omnidata --partition learnlab \
#     --dataset_size 1_45m

# python submitit_pretrain.py \
#     --wandb_name osd_1_45m_vit_base_01 \
#     --nodes 8 \
#     --batch_size 128 \
#     --model mae_vit_base_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 800 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path /checkpoint/karmeshyadav/omnidataset/ \
#     --dataset_type omnidata --partition learnlab \
#     --dataset_size 1_45m \
#     --use_volta32

# python submitit_pretrain.py \
#     --wandb_name mae_vit_small_01 \
#     --nodes 2 \
#     --batch_size 256 \
#     --accum_iter 1 \
#     --model mae_vit_small_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 400 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
#     --output_dir /checkpoint/karmeshyadav/mae_training/ \
#     --partition learnlab --use_volta32

# python submitit_pretrain.py \
#     --wandb_name mae_vit_small_02 \
#     --nodes 1 \
#     --batch_size 512 \
#     --accum_iter 1 \
#     --model mae_vit_small_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 800 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
#     --output_dir /checkpoint/karmeshyadav/mae_training/ \
#     --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name mae_vit_base_01 \
    --nodes 4 \
    --batch_size 128 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition devlab --use_volta32

python submitit_pretrain.py \
    --wandb_name mae_vit_base_02 \
    --nodes 4 \
    --batch_size 128 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32