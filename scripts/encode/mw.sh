python encode_dataset.py -m \
    task=mw-drawer-close,mw-drawer-open,mw-hammer,mw-box-close,mw-push \
    suite=mw \
    setting=offline \
    modality=pixels \
    +features=maehoi \
    +use_all=true \
    hydra/launcher=slurm \
    hydra.job.name=mw-enc \
    hydra.launcher.timeout_min=300 \
    hydra.launcher.mem_gb=48
