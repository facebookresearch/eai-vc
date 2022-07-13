CUDA_VISIBLE_DEVICES=1 python train_renderer.py \
    task=finger-spin \
    modality=pixels \
    encoder.arch=default+ \
    +tdmpc_artifact=finger-spin-pixels-offline-v1-1-chkpt:v0 \
    fraction=0.005 \
    hydra/launcher=slurm
