python train_offline.py -m \
    task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-push \
    suite=mw \
    setting=offline \
    modality=state \
    exp_name=offline-v1 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mw-state \
    hydra.launcher.timeout_min=800 \
    hydra.launcher.mem_gb=32
