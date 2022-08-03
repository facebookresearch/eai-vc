python train.py -m \
    task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=online \
    modality=map \
    features=mocoego18 \
    feature_dims=[32,14,14] \
    pool_fn=max \
    exp_name=v2-map2-maxpool8x \
    seed=1,2 \
    hydra/launcher=slurm \
    hydra.job.name=m2max8x \
    hydra.launcher.timeout_min=1800 \
    hydra.launcher.mem_gb=64
