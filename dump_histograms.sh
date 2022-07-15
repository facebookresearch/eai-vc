python train_offline.py \
    -m task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-push,mw-pick-place \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    action_repeat=2 \
    episode_length=250 \
    exp_name=test \
    +dump_histogram=true \
    seed=1 \
    hydra/launcher=local
