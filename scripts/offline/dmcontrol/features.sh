python train_offline.py -m \
    task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=offline \
    modality=features \
    features=maehoi \
    exp_name=offline-v2 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-features \
    hydra.launcher.timeout_min=1200 \
    hydra.launcher.mem_gb=48
