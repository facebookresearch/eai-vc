python evaluate_open_loop.py \
    task=quadruped-run \
    modality=features \
    features=mocoego \
    +target_modality=pixels \
    horizon=500 \
    iterations=16 \
    num_samples=8192 \
    exp_name=offline-v1 \
    seed=2 \
    hydra/launcher=slurm
