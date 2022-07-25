python generate.py \
    -m task=$DMCONTROL \
    suite=dmcontrol \
    setting=online \
    modality=state \
    exp_name=v1 \
    seed=1,2,3 \
    +identifier_id=0,1,2,3,4,5,6,7,8,9,10 \
    hydra/launcher=slurm \
    hydra.job.name=dmc-gen \
    hydra.launcher.timeout_min=300 \
    hydra.launcher.mem_gb=32
