python generate.py \
    -m task=$MW \
    suite=mw \
    setting=online \
    modality=state \
    +identifier_id=0,1,2,3,4,5,6,7,8,9,10 \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mw-gen \
    hydra.launcher.timeout_min=500 \
    hydra.launcher.mem_gb=48
