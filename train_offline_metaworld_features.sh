python train_offline.py \
    -m task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-pick-place,mw-shelf-place \
    modality=features \
    features=mocoego,random \
    frame_stack=1 \
    action_repeat=2 \
    episode_length=250 \
    exp_name=offline-v1-2xdata \
    seed=1,2,3 \
    hydra/launcher=slurm
