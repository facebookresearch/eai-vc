python generate.py \
    -m task=mw-shelf-place \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    per=true \
    action_repeat=2 \
    episode_length=250 \
    train_steps=250000 \
    eval_freq=50000 \
    save_freq=50000 \
    exp_name=v1 \
    seed=1 \
    hydra/launcher=slurm
