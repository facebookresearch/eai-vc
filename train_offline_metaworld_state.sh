python train_offline.py \
    -m task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-pick-place,mw-shelf-place \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    action_repeat=2 \
    episode_length=250 \
    exp_name=offline-v1-100p \
    seed=1,2,3 \
    hydra/launcher=slurm
