python train.py -m \
    task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-push \
    suite=mw \
    setting=online \
    modality=state \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mw-state \
    hydra.launcher.timeout_min=1300 \
    hydra.launcher.mem_gb=32
