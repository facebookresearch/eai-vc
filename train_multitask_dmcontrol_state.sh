python train_offline.py \
    task=dmcontrol-mt5 \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    train_iter=500000 \
    exp_name=v1 \
    seed=1 \
    hydra/launcher=slurm
