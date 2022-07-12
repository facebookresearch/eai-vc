python train_offline.py \
    task=mw-mt5 \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    action_repeat=2 \
    episode_length=250 \
    train_iter=500000 \
    exp_name=v1 \
    seed=1 \
    goal_hidden=true \
    hydra/launcher=slurm
