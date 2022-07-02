TASKS=finger-spin,finger-turn-easy,finger-turn-hard,cup-catch,reacher-easy,reacher-hard

python generate.py -m task=$TASKS enc_dim=256 mlp_dim=512 per=true seed=1,2,3 hydra/launcher=slurm
