python train_offline.py \
    -m task=mw-mt15 \
    modality=features \
    features=mocoego \
    frame_stack=1 \
    batch_size=2048 \
    episode_length=250 \
    eval_freq=25000 \
    train_iter=300000 \
    exp_name=offline-v1-b2048 \
    seed=1,2 \
    hydra/launcher=slurm
