python train_offline.py \
    -m task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-push,mw-pick-place \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    action_repeat=1 \
    episode_length=500 \
    exp_name=offline-v3 \
    seed=1,2,3 \
    hydra/launcher=slurm
