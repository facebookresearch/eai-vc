TASKS=cartpole-swingup,cartpole-balance,pendulum-swingup

python generate.py -m task=$TASKS enc_dim=256 mlp_dim=512 per=true seed=1,2,3 hydra/launcher=slurm
