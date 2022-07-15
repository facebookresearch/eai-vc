python train_renderer.py \
    -m task=walker-run \
    modality=features \
    features=random \
    +target_modality=state \
    batch_size=128 \
    train_iter=10000 \
    eval_freq=200 \
    save_freq=5000 \
    exp_name=offline-v1 \
    seed=1,2,3 \
    +use_val=true \
    +all_modalities=true \
    hydra/launcher=slurm
