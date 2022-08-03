python train.py -m \
    task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-push \
    suite=mw \
    setting=online \
    modality=features \
    features=mocometaworld,mocoego,random \
    exp_name=v2-fs \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mw-features \
    hydra.launcher.timeout_min=1400 \
    hydra.launcher.mem_gb=48
