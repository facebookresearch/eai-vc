python train_offline.py \
    -m task=$MT15 \
    suite=mw \
    setting=multitask \
    modality=features \
    features=mocoego,random \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mt15-features \
    hydra.launcher.timeout_min=900 \
    hydra.launcher.mem_gb=448
