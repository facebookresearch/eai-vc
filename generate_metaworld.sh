python generate.py \
    task=mw-box-close \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    per=true \
    episode_length=250 \
    train_steps=500000 \
    eval_freq=50000 \
    save_freq=50000 \
    +identifier_id=10 \
    exp_name=v1 \
    seed=1 \
    hydra/launcher=slurm
