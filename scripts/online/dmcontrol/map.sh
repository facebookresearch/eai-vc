python train.py -m \
    task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=online \
    modality=map \
    features=mocoego18 \
    exp_name=v2-map2-aug-simple \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-map \
    hydra.launcher.timeout_min=2100 \
    hydra.launcher.mem_gb=76
