python train_renderer.py -m \
    task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-push \
    suite=mw \
    setting=sim \
    modality=features \
    target_modality=state \
    features=mocometaworld,mocoego,random \
    exp_name=offline-v2 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mw-sim \
    hydra.launcher.timeout_min=300 \
    hydra.launcher.mem_gb=48
