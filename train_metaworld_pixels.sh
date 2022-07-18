python train.py \
    -m task=mw-reach,mw-push,mw-assembly,mw-soccer,mw-faucet-close,mw-faucet-open,mw-door-open,mw-door-close,mw-window-open,mw-window-close \
    modality=pixels \
    enc_dim=256 \
    mlp_dim=512 \
    frame_stack=1 \
    episode_length=250 \
    train_steps=500000 \
    eval_freq=50000 \
    save_freq=50000 \
    exp_name=v1 \
    seed=1,2 \
    hydra/launcher=slurm
