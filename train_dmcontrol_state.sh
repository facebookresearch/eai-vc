python train.py \
    -m task=cup-catch,finger-spin,cheetah-run,walker-run,quadruped-run \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    per=true \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm
