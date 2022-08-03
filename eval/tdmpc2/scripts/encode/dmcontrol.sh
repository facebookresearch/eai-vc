python encode_dataset.py -m \
    task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=offline \
    modality=pixels \
    +features=maehoi \
    +use_all=true \
    hydra/launcher=slurm \
    hydra.job.name=dmc-enc \
    hydra.launcher.timeout_min=300 \
    hydra.launcher.mem_gb=48
