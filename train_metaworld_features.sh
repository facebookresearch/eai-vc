python train.py \
    -m task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-pick-place,mw-shelf-place \
    modality=features \
    features=mocoego,random \
    enc_dim=256 \
    mlp_dim=512 \
    per=true \
    frame_stack=1 \
    action_repeat=2 \
    episode_length=250 \
    train_steps=500000 \
    eval_freq=50000 \
    save_freq=50000 \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm
