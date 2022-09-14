## Pre-training MAE

To pre-train ViT-Base (recommended default) with **multi-node distributed training**, run the following on 8 nodes with 8 GPUs each:
```
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
```
- Here the effective batch size is 64 (`batch_size` per gpu) * 8 (`nodes`) * 8 (gpus per node) = 4096. If memory or # gpus is limited, use `accum_iter` to maintain the effective batch size, which is `batch_size` (per gpu) * `nodes` * 8 (gpus per node) * `accum_iter`.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- Here we use `norm_pix_loss` as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off `norm_pix_loss` using `norm_pix_loss=False`.
- The exact same hyper-parameters and configs (initialization, augmentation, etc.) are used as our TF/TPU implementation. In our sanity checks, this PT/GPU re-implementation can reproduce the TF/TPU results within reasonable random variation. We get 85.5% [fine-tuning](FINETUNE.md) accuracy by pre-training ViT-Large for 800 epochs (85.4% in paper Table 1d with TF/TPU).
- Training time is ~42h in 64 V100 GPUs (800 epochs).

**[New]**
- Use the argument `wandb.name` to give a name to your folder created locally and to the run on wandb
- We can pretrain a combination of 3 datasets: `imagenet`, `omnidata` and `hm3d+gibson`, by passing the correct argument through `dataset_type`
- `data_path` is the location of the folder containing txt files with the path of all the images. This allows much faster starting of training compared to `torchvision.datasets.ImageFolder`.

To train ViT-Small or ViT-Large, set `mae_model=mae_vit_small_patch16` or `mae_model=mae_vit_large_patch14`.
