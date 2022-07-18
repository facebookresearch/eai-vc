python encode_dataset.py \
    -m task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-reach,mw-push,mw-pick-place,mw-assembly,mw-soccer,mw-faucet-close,mw-faucet-open,mw-door-open,mw-door-close,mw-window-open,mw-window-close \
    modality=pixels \
    +features=mocoego,random,mocometaworld \
    episode_length=250 \
    +use_all=true \
    hydra/launcher=slurm
