python train_offline.py -m \
    task=mw-mt15 \
    suite=mw \
    setting=multitask \
    modality=state \
    exp_name=offline-v1 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mt15-state \
    hydra.launcher.timeout_min=900 \
    hydra.launcher.mem_gb=64
