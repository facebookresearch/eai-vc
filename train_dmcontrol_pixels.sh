python train.py \
    -m task=walker-walk \
    modality=pixels \
    enc_dim=256 \
    mlp_dim=512 \
    per=true \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm
