python train_offline.py \
    -m task=$DMCONTROL \
    suite=dmcontrol \
    setting=offline \
    modality=features \
    features=mocoego,random \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-features \
    hydra.launcher.timeout_min=1000 \
    hydra.launcher.mem_gb=48
