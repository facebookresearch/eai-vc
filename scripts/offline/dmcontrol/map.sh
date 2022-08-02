python train_offline.py -m \
    task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=offline \
    modality=map \
    features=mocoego18 \
    exp_name=v2-map2-aug-simple \
    seed=1,2 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-map \
    hydra.launcher.timeout_min=1700 \
    hydra.launcher.mem_gb=180
