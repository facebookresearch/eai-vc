python train_offline.py \
    -m task=mw-box-close \
    modality=state \
    episode_length=250 \
    exp_name=offline-v1-again-per \
    seed=1,2 \
    hydra/launcher=slurm
