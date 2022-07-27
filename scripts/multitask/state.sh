python train_offline.py -m \
    task=mw-mt10 \
    suite=mw \
    setting=multitask \
    modality=state \
    exp_name=offline-v2 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mt10-state \
    hydra.launcher.timeout_min=1100 \
    hydra.launcher.mem_gb=32 \
    hydra.launcher.cpus_per_task=10
