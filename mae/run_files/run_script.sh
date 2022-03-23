python submitit_pretrain.py \
    --wandb_name imagenet_mae_reproduce \
    --nodes 8 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/imagenet_full_size/ \
    --dataset_type imagenet 

python submitit_pretrain.py \
    --wandb_name osd_1_45_default_params_01 \
    --nodes 8 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/omnidataset/ \
    --dataset_type omnidata --partition learnlab \
    --dataset_size 1_45m

python submitit_pretrain.py \
    --wandb_name imagenet_mae_reproduce_01 \
    --nodes 8 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/imagenet_full_size/ \
    --dataset_type imagenet --partition learnlab