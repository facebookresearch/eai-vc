python train_offline.py \
    task=pendulum-swingup \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    exp_name=test \
    +dump_histogram=true \
    seed=1 \
    hydra/launcher=local
