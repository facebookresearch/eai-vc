python train_offline.py -m \
    task=mw-mt10 \
    suite=mw \
    setting=multitask \
    algorithm=mtdmpc \
    modality=pixels \
    exp_name=offline-v2-prop \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mtdmpc-pixels \
    hydra.launcher.timeout_min=2900 \
    hydra.launcher.mem_gb=80
    