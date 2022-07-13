python train_renderer.py \
    task=cup-catch \
    modality=pixels \
    encoder.arch=default+ \
    enc_dim=256 \
    mlp_dim=512 \
    +tdmpc_artifact=cup-catch-pixels-v1-3-500000:v0 \
    fraction=0.01 \
    hydra/launcher=slurm
