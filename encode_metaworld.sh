python encode_dataset.py \
    -m task=mw-shelf-place \
    modality=pixels \
    +features=mocometaworld \
    action_repeat=2 \
    episode_length=250 \
    hydra/launcher=slurm
