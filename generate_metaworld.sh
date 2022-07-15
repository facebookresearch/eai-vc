python generate.py \
    -m task=mw-reach,mw-faucet-close,mw-faucet-open,mw-assembly,mw-soccer \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    per=true \
    action_repeat=2 \
    episode_length=250 \
    train_steps=500000 \
    eval_freq=50000 \
    save_freq=50000 \
    +identifier_id=0,1,2,3,4,5,6,7,8,9,10 \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm
