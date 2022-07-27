python train_offline.py -m \
    task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-push \
    suite=mw \
    setting=offline \
    modality=features \
    features=mocometaworld,random \
    exp_name=offline-v2 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mw-features \
    hydra.launcher.timeout_min=1200 \
    hydra.launcher.mem_gb=48
