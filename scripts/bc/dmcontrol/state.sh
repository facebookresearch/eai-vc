python train_offline.py -m \
    task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=offline \
    modality=state \
    algorithm=bc \
    exp_name=offline-v2 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-state \
    hydra.launcher.timeout_min=800 \
    hydra.launcher.mem_gb=32
