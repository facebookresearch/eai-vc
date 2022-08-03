python train_renderer.py -m \
    task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=sim \
    modality=state \
    target_modality=state \
    +img_size=84 \
    exp_name=offline-v2 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=sim-state \
    hydra.launcher.timeout_min=300 \
    hydra.launcher.mem_gb=32
