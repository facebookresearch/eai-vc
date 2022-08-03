python train_offline.py -m \
    task=mw-mt10 \
    suite=mw \
    setting=multitask \
    modality=pixels \
    exp_name=offline-v2 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mt10-pixels \
    hydra.launcher.timeout_min=1900 \
    hydra.launcher.mem_gb=64
    