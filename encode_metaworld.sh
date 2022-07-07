python encode_dataset.py \
    -m task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-pick-place,mw-shelf-place \
    modality=pixels \
    +features=mocodmcontrol,mocoego,random \
    hydra/launcher=slurm
