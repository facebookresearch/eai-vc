python train_renderer.py -m \
    task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-push \
    suite=mw \
    setting=sim \
    modality=pixels \
    target_modality=pixels \
    eval_freq=1000 \
    save_freq=5000 \
    exp_name=offline-v2 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mw-sim \
    hydra.launcher.timeout_min=1000 \
    hydra.launcher.mem_gb=96
