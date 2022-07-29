python evaluate_open_loop.py \
    task=mw-drawer-close \
    suite=mw \
    setting=offline \
    modality=features \
    features=mocoego \
    +target_modality=pixels \
    horizon=500 \
    iterations=16 \
    num_samples=8192 \
    exp_name=offline-v2 \
    seed=1