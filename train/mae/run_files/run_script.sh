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
    --nodes 2 \
    --batch_size 256 \
    --model mae_vit_base_patch16 \
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
    --nodes 2 \
    --batch_size 256 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name mae_vit_base_01_cj \
    --nodes 2 \
    --batch_size 256 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32 --color_jitter

python submitit_pretrain.py \
    --wandb_name mae_vit_small_masking_0_95_viz \
    --nodes 2 \
    --batch_size 256 \
    --accum_iter 1 \
    --model mae_vit_small_patch16 \
    --mask_ratio 0.95 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    --wandb_name mae_vit_small_masking_0_75_viz \
    --nodes 2 \
    --batch_size 256 \
    --accum_iter 1 \
    --model mae_vit_small_patch16 \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition devlab --use_volta32

python submitit_pretrain.py \
    --wandb_name mae_vit_small_02_cj \
    --nodes 2 \
    --batch_size 256 \
    --model mae_vit_small_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32 --color_jitter

python submitit_pretrain.py \
    --wandb_name mae_vit_small_decoder_large_HGPS_RE10K \
    --nodes 2 \
    --batch_size 256 \
    --model mae_vit_small_patch16_large_decoder \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train/ /checkpoint/karmeshyadav/real-estate-10k-frames-v0/ \
    --output_dir /checkpoint/karmeshyadav/mae_training/ \
    --partition learnlab --use_volta32

python submitit_pretrain.py \
    wandb.name=mae_vit_base_HGSP \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=800 \
    warmup_epochs=40 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    data_path=[/checkpoint/yixinlin/eaif/datasets/hm3d+gibson/v1/train] \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

python submitit_pretrain.py \
    wandb.name=mae_vit_small_HGSP \
    batch_size=256 \
    mae_model=mae_vit_small_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=800 \
    warmup_epochs=40 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    data_path=[/checkpoint/yixinlin/eaif/datasets/hm3d+gibson/v1/train] \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

python submitit_pretrain.py \
    wandb.name=mae_vit_base_HGSP_Ego4D \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=800 \
    warmup_epochs=40 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    data_path=[/checkpoint/yixinlin/eaif/datasets/hm3d+gibson/v1/train,/checkpoint/yixinlin/eaif/datasets/ego4d] \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

python submitit_pretrain.py \
    wandb.name=mae_vit_small_HGSP_Ego4D \
    batch_size=2 \
    mae_model=mae_vit_small_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=800 \
    warmup_epochs=40 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    data_path=[/checkpoint/yixinlin/eaif/datasets/hm3d+gibson/v1/train,/checkpoint/yixinlin/eaif/datasets/ego4d] \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

python submitit_pretrain.py \
    wandb.name=mae_vit_base_Ego4D \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=800 \
    warmup_epochs=40 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    data_path=[/checkpoint/yixinlin/eaif/datasets/ego4d] \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

python submitit_pretrain.py \
    wandb.name=mae_vit_small_Ego4D \
    batch_size=512 \
    mae_model=mae_vit_small_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=800 \
    warmup_epochs=40 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    data_path=[/checkpoint/yixinlin/eaif/datasets/ego4d] \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=1 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 6,216,416 Ego4D + HGSP * 800 = 165 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_base_HGSP_Ego4D_165_epochs \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=165 \
    warmup_epochs=8 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    data_path=[/checkpoint/yixinlin/eaif/datasets/hm3d+gibson/v1/train,/checkpoint/yixinlin/eaif/datasets/ego4d] \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 4,874,498 Ego4D * 800 = 210 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_base_Ego4D_210_epochs \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=210 \
    warmup_epochs=10 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    data_path=[/checkpoint/yixinlin/eaif/datasets/ego4d] \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 2,468,508 ego_inav_subset * 800 = 415 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_base_ego_inav_subset \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=415 \
    warmup_epochs=21 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego_inav_subset \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 4,377,319 ego_inav * 800 = 234 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_base_ego_inav \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=234 \
    warmup_epochs=12 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego_inav \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 2,468,508 ego_inav_subset * 800 = 415 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_large_ego_inav_subset \
    batch_size=128 \
    mae_model=mae_vit_large_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=415 \
    warmup_epochs=21 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego_inav_subset \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=4 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 4,377,319 ego_inav * 800 = 234 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_large_ego_inav \
    batch_size=128 \
    mae_model=mae_vit_large_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=234 \
    warmup_epochs=12 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego_inav \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=4 \
    partition=learnlab \
    use_volta32=True

python submitit_pretrain.py \
    wandb.name=mae_vit_base_ego_bigger_inav \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=800 \
    warmup_epochs=40 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego_mvp_freq_inav \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

python submitit_pretrain.py \
    wandb.name=mae_vit_large_ego_bigger_inav \
    batch_size=128 \
    mae_model=mae_vit_large_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=800 \
    warmup_epochs=40 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego_mvp_freq_inav \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=4 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 4,340,820 ego_bigger_inav * 800 = 236 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_large_ego_bigger_inav_236_epochs \
    batch_size=128 \
    mae_model=mae_vit_large_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=236 \
    warmup_epochs=12 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego_mvp_freq_inav \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=4 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 4,340,820 ego_bigger_inav * 800 = 236 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_base_ego_bigger_inav_236_epochs \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=236 \
    warmup_epochs=12 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego_mvp_freq_inav \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 802,529 inav * 800 = 1277 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_base_inav_1277_epochs \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=1277 \
    warmup_epochs=63 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=inav \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 3,538,291 ego * 800 = 289 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_base_ego_289_epochs \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=289 \
    warmup_epochs=14 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego3_5m \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 3,538,291 ego * 800 = 289 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_large_ego_289_epochs \
    batch_size=128 \
    mae_model=mae_vit_large_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=289 \
    warmup_epochs=14 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego3_5m \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=4 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 5,621,987 ego+inav+imgnet * 800 = 182 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_base_ego_inav_imgnet_182_epochs \
    batch_size=256 \
    mae_model=mae_vit_base_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=182 \
    warmup_epochs=9 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego_inav_imgnet \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=2 \
    partition=learnlab \
    use_volta32=True

# 1,281,167 Imgnet / 5,621,987 ego+inav+imgnet * 800 = 182 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_large_ego_inav_imgnet_182_epochs \
    batch_size=128 \
    mae_model=mae_vit_large_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=182 \
    warmup_epochs=9 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego_inav_imgnet \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=4 \
    partition=learnlab \
    use_volta32=True


# 1,281,167 Imgnet / 50,992,643 ego51 * 800 = 40 Epochs
python submitit_pretrain.py \
    wandb.name=mae_vit_large_ego51_epochs \
    batch_size=128 \
    mae_model=mae_vit_large_patch16 \
    norm_pix_loss=True \
    mask_ratio=0.75 \
    epochs=40 \
    warmup_epochs=2 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    dataset=ego51m \
    output_dir=/checkpoint/yixinlin/eaif/results/mae_training/ \
    nodes=4 \
    partition=learnlab \
    use_volta32=True