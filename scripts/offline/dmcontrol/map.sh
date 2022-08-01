python train_offline.py -m \
    task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=offline \
    modality=pixels \
    encoder=mocoego \
    +feature_map=2 \
    exp_name=mocoego-offline-v2 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-pixels \
    hydra.launcher.timeout_min=1700 \
    hydra.launcher.mem_gb=48
