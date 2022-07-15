python evaluate_open_loop.py \
    task=quadruped-run \
    modality=features \
    features=mocoego \
    +target_modality=pixels \
    horizon=10 \
    iterations=12 \
    num_samples=4096 \
    exp_name=offline-v1 \
    seed=2 \
    hydra/launcher=slurm
