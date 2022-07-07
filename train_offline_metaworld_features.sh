python train_offline.py \
    task=mw-drawer-close \
    modality=features \
    features=mocoego \
    frame_stack=1 \
    action_repeat=2 \
    episode_length=250 \
    exp_name=test \
    seed=1 \
    hydra/launcher=slurm
