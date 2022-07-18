python train_offline.py \
    -m task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-reach,mw-push,mw-pick-place,mw-assembly,mw-soccer,mw-faucet-close,mw-faucet-open,mw-door-open,mw-door-close,mw-window-open,mw-window-close \
    modality=features \
    features=mocoego,random \
    frame_stack=1 \
    episode_length=250 \
    exp_name=offline-v1-again-per \
    seed=1,2,3 \
    hydra/launcher=slurm
