TASKS=finger-spin,finger-turn-easy,finger-turn-hard,cup-catch,reacher-easy,reacher-hard

python train.py -m task=$TASKS modality=state enc_dim=256 mlp_dim=512 per=true exp_name=v1 seed=1,2,3 hydra/launcher=slurm
