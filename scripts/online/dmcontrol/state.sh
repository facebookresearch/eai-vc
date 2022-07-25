python train.py -m \
    -m task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=online \
    modality=state \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-state \
    hydra.launcher.timeout_min=900 \
    hydra.launcher.mem_gb=24
