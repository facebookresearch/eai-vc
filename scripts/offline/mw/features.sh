python train_offline.py \
    -m task=$MW \
    suite=mw \
    setting=offline \
    modality=features \
    features=mocoego,random \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mw-features \
    hydra.launcher.timeout_min=1000 \
    hydra.launcher.mem_gb=48
