python train_renderer.py \
    task=mw-drawer-open \
    modality=pixels \
    encoder.arch=default+ \
    batch_size=128 \
    frame_stack=1 \
    action_repeat=2 \
    episode_length=250 \
    train_iter=10000 \
    eval_freq=200 \
    save_freq=5000 \
    exp_name=offline-v1 \
    seed=1 \
    fraction=0.01 \
    +use_val=true \
    hydra/launcher=slurm
