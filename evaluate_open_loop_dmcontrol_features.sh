python evaluate_open_loop.py \
    task=cup-catch \
    modality=features \
    features=mocoego \
    +target_modality=pixels \
    horizon=50 \
    iterations=16 \
    num_samples=4098 \
    exp_name=offline-v1 \
    seed=1 \
    hydra/launcher=slurm
