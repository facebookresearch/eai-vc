python train_offline.py -m \
    task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-push \
    suite=mw \
    setting=offline \
    modality=pixels \
    exp_name=offline-v2-fs \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mw-pixels \
    hydra.launcher.timeout_min=1200 \
    hydra.launcher.mem_gb=48
