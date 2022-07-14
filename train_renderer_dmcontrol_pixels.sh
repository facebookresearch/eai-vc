python train_renderer.py \
    -m task=cup-catch,finger-spin,cheetah-run,walker-run,quadruped-run \
    modality=pixels \
    encoder.arch=default+ \
    +target_modality=pixels \
    batch_size=128 \
    train_iter=10000 \
    eval_freq=200 \
    save_freq=5000 \
    exp_name=offline-v1 \
    seed=1,2,3 \
    +use_val=true \
    hydra/launcher=slurm
