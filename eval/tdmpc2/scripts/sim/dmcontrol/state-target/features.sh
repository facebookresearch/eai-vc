python train_renderer.py -m \
    task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=sim \
    modality=features \
    target_modality=state \
    features=mocodmcontrol,mocoego,random \
    exp_name=offline-v2 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-sim \
    hydra.launcher.timeout_min=300 \
    hydra.launcher.mem_gb=48
