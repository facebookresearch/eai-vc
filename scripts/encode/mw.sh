python encode_dataset.py \
    -m task=$MW \
    suite=mw \
    setting=online \
    modality=pixels \
    +features=mocoego,random \
    +use_all=true \
    hydra/launcher=slurm \
    hydra.job.name=mw-enc \
    hydra.launcher.timeout_min=300 \
    hydra.launcher.mem_gb=48
