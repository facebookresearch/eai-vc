python train_offline.py -m \
    task=mw-mt10 \
    suite=mw \
    setting=multitask \
    algorithm=mtdmpc \
    modality=features \
    features=mocoego \
    exp_name=offline-v2-mtdmpc \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mtdmpc-features \
    hydra.launcher.timeout_min=2600 \
    hydra.launcher.mem_gb=64 \
    hydra.launcher.cpus_per_task=10
