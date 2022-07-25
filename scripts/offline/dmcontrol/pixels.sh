python train_offline.py \
    -m task=$DMCONTROL \
    suite=dmcontrol \
    setting=offline \
    modality=pixels \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-pixels \
    hydra.launcher.timeout_min=1000 \
    hydra.launcher.mem_gb=48
