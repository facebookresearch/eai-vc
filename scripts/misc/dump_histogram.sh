python train_offline.py -m \
    task=cup-catch,finger-spin,cheetah-run,walker-walk,walker-run \
    suite=dmcontrol \
    setting=offline \
    modality=state \
    exp_name=test \
    +dump_histogram=true \
    seed=1 \
    hydra/launcher=slurm \
    hydra.job.name=histogram \
    hydra.launcher.timeout_min=30 \
    hydra.launcher.mem_gb=24
