python train_offline.py \
    task=mw-box-close \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    action_repeat=2 \
    episode_length=250 \
    exp_name=test \
    seed=1 \
    hydra/launcher=slurm
