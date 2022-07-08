python train_offline.py \
    task=* \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    exp_name=test \
    seed=1 \
    hydra/launcher=slurm
