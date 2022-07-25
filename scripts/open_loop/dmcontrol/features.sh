python evaluate_open_loop.py \
    -m task=$DMCONTROL \
    suite=dmcontrol \
    setting=offline \
    modality=features \
    features=mocoego \
    +target_modality=pixels \
    horizon=500 \
    iterations=16 \
    num_samples=8192 \
    exp_name=offline-v1 \
    seed=1,2,3 \
    hydra/launcher=slurm
