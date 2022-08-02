python train.py -m \
    task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-push \
    suite=mw \
    setting=online \
    modality=map \
    features=mocoego18 \
    exp_name=v2-map3-aug-simple \
    seed=1,2 \
    hydra/launcher=slurm \
    hydra.job.name=mw-map \
    hydra.launcher.timeout_min=2100 \
    hydra.launcher.mem_gb=76
