python encode_dataset.py \
    -m task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-pick-place,mw-shelf-place \
    modality=pixels \
    +features=mocoego,random \
    action_repeat=2 \
    episode_length=250 \
    +use_all=true \
    hydra/launcher=slurm
