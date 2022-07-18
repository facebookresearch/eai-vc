python train_offline.py \
    -m task=mw-mt15 \
    modality=state \
    batch_size=2048 \
    enc_dim=1024 \
    episode_length=250 \
    train_iter=400000 \
    exp_name=offline-v1-taskenc-b2048-e1024 \
    seed=1,2 \
    hydra/launcher=slurm
