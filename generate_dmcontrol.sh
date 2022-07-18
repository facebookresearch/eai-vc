python generate.py \
    -m task=pendulum-swingup \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    per=true \
    exp_name=v1 \
    seed=1,2,3 \
    +identifier_id=0,1,2,3,4,5,6,7,8,9,10 \
    hydra/launcher=slurm
