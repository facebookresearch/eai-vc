python train.py -m \
    task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=online \
    modality=pixels \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-pixels \
    hydra.launcher.timeout_min=1700 \
    hydra.launcher.mem_gb=56
