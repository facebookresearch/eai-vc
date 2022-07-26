python train_offline.py -m \
    task=cheetah-run,walker-run \
    suite=dmcontrol \
    setting=offline \
    modality=state \
    +lazy_load=true \
    exp_name=offline-v2-lazyload \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-state \
    hydra.launcher.timeout_min=900 \
    hydra.launcher.mem_gb=32
