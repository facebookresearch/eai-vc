python train.py \
    -m task=mw-reach,mw-window-close,mw-window-open,mw-door-close,mw-door-open,mw-assembly,mw-soccer,mw-faucet-close,mw-faucet-open \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    per=true \
    action_repeat=2 \
    episode_length=250 \
    train_steps=500000 \
    eval_freq=50000 \
    save_freq=50000 \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm
