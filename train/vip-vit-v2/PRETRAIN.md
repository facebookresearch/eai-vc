## Pre-training MAE

To pre-train VIP-ViT models on Ego4D: 

```
python submitit_pretrain.py \
    wandb.name=vip_vit_small_Ego4D \
    batch_size=64 \
    mae_model=mae_vit_small_patch16 \
    vip=True \
    epochs=220 \
    warmup_epochs=20 \
    blr=1e-6 \
    weight_decay=0.05 \
    data_path=[/checkpoint/maksymets/eaif/datasets/ego4d] \
    output_dir=/checkpoint/maksymets/eaif/results/vip_training/ \
    nodes=8 \
    partition=learnlab \
    use_volta32=True
```
```
python submitit_pretrain.py \
    wandb.name=vip_vit_base_Ego4D \
    batch_size=64 \
    mae_model=mae_vit_base_patch16 \
    vip=True \
    epochs=220 \
    warmup_epochs=20 \
    blr=1e-6 \
    weight_decay=0.05 \
    data_path=[/checkpoint/maksymets/eaif/datasets/ego4d] \
    output_dir=/checkpoint/maksymets/eaif/results/vip_training/ \
    nodes=8 \
    partition=learnlab \
    use_volta32=True
```

Ego4D+HGSP:
```
python submitit_pretrain.py \
    wandb.name=vip_vit_base_Ego4D_HGSP \
    batch_size=64 \
    mae_model=mae_vit_base_patch16 \
    vip=True \
    epochs=170 \
    warmup_epochs=20 \
    blr=1e-6 \
    weight_decay=0.05 \
    data_path=[/checkpoint/maksymets/eaif/datasets/hm3d+gibson/v1/train,/checkpoint/maksymets/eaif/datasets/ego4d] \
    output_dir=/checkpoint/maksymets/eaif/results/vip_training/ \
    nodes=8 \
    partition=learnlab \
    use_volta32=True
```
```
python submitit_pretrain.py \
    wandb.name=vip_vit_small_Ego4D_HGSP \
    batch_size=64 \
    mae_model=mae_vit_small_patch16 \
    vip=True \
    epochs=170 \
    warmup_epochs=20 \
    blr=1e-6 \
    weight_decay=0.05 \
    data_path=[/checkpoint/maksymets/eaif/datasets/hm3d+gibson/v1/train,/checkpoint/maksymets/eaif/datasets/ego4d] \
    output_dir=/checkpoint/maksymets/eaif/results/vip_training/ \
    nodes=8 \
    partition=learnlab \
    use_volta32=True
```

HGSP:
```
python submitit_pretrain.py \
    wandb.name=vip_vit_base_HGSP \
    batch_size=64 \
    mae_model=mae_vit_base_patch16 \
    vip=True \
    epochs=800 \
    warmup_epochs=20 \
    blr=1e-6 \
    weight_decay=0.05 \
    resume=/checkpoint/maksymets/eaif/results/vip_training/vip_vit_base_HGSP/2022-10-26_21-24-04/checkpoint-400.pth \
    data_path=[/checkpoint/maksymets/eaif/datasets/hm3d+gibson/v1/train] \
    output_dir=/checkpoint/maksymets/eaif/results/vip_training/ \
    nodes=8 \
    partition=learnlab \
    use_volta32=True
```

Fine-tuning on top of a trained MAE:
```
python submitit_pretrain.py \
    wandb.name=vip_vit_small_Ego4D_finetune_mae \
    batch_size=64 \
    mae_model=mae_vit_small_patch16 \
    vip=True \
    epochs=800 \
    warmup_epochs=20 \
    blr=1.5e-4 \
    weight_decay=0.05 \
    resume=/checkpoint/maksymets/eaif/models/mae_ego4d/mae_vit_small_patch16_ego4d_800_epochs.pth \
    data_path=[/checkpoint/maksymets/eaif/datasets/ego4d] \
    output_dir=/checkpoint/maksymets/eaif/results/vip_training/ \
    nodes=8 \
    partition=learnlab \
    use_volta32=True
```

Use global_pool version: 
```
python submitit_pretrain.py \
    wandb.name=vip_vit_small_Ego4D_global_pool \
    batch_size=64 \
    mae_model=mae_vit_small_patch16 \
    vip=True use_cls=False global_pool=True \
    epochs=800 \
    warmup_epochs=20 \
    blr=1.5e-6 \
    weight_decay=0.05 \
    data_path=[/checkpoint/maksymets/eaif/datasets/ego4d] \
    output_dir=/checkpoint/maksymets/eaif/results/vip_training/ \
    nodes=8 \
    partition=learnlab \
    use_volta32=True
```
```
python submitit_pretrain.py \
    wandb.name=vip_vit_base_Ego4D_global_pool \
    batch_size=64 \
    mae_model=mae_vit_base_patch16 \
    vip=True use_cls=False global_pool=True \
    epochs=800 \
    warmup_epochs=20 \
    blr=1.5e-6 \
    weight_decay=0.05 \
    data_path=[/checkpoint/maksymets/eaif/datasets/ego4d] \
    output_dir=/checkpoint/maksymets/eaif/results/vip_training/ \
    nodes=8 \
    partition=learnlab \
    use_volta32=True
```
## Variants that haven't worked yet (?)
Use Mask: 
```
python submitit_pretrain.py \
    wandb.name=vip_vit_base_Ego4D_mask \
    batch_size=64 \
    mae_model=mae_vit_base_patch16 \
    use_mask=True \
    vip=True \
    epochs=220 \
    warmup_epochs=20 \
    blr=1.5e-6 \
    weight_decay=0.05 \
    data_path=[/checkpoint/maksymets/eaif/datasets/ego4d] \
    output_dir=/checkpoint/maksymets/eaif/results/vip_training/ \
    nodes=8 \
    partition=devlab \
    use_volta32=True
```
```
python submitit_pretrain.py \
    wandb.name=vip_vit_small_Ego4D_mask_finetune \
    batch_size=128 \
    mae_model=mae_vit_small_patch16 \
    use_mask=True \
    vip=True \
    epochs=220 \
    warmup_epochs=20 \
    blr=1.5e-6 \
    weight_decay=0.05 \
    resume=/checkpoint/maksymets/eaif/models/mae_ego4d/mae_vit_small_patch16_ego4d_800_epochs.pth \
    data_path=[/checkpoint/maksymets/eaif/datasets/ego4d] \
    output_dir=/checkpoint/maksymets/eaif/results/vip_training/ \
    nodes=4 \
    partition=devlab \
    use_volta32=True
```
```
python submitit_pretrain.py \
    wandb.name=vip_vit_small_Ego4D_mask \
    batch_size=128 \
    mae_model=mae_vit_small_patch16 \
    use_mask=True \
    vip=True \
    epochs=220 \
    warmup_epochs=20 \
    blr=1.5e-6 \
    weight_decay=0.05 \
    data_path=[/checkpoint/maksymets/eaif/datasets/ego4d] \
    output_dir=/checkpoint/maksymets/eaif/results/vip_training/ \
    nodes=4 \
    partition=devlab \
    use_volta32=True
```