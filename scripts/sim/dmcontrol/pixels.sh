python train_renderer.py \
    -m task=$DMCONTROL \
    suite=dmcontrol \
    setting=sim \
    modality=pixels \
    target_modality=state \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=sim-pixels \
    hydra.launcher.timeout_min=1000 \
    hydra.launcher.mem_gb=96
