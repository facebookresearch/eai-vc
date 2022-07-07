# Full task sets
# WALKER=walker-walk,walker-run,walker-stand,walker-walk-backwards,walker-run-backwards,walker-arabesque,walker-lie-down,walker-legs-up,walker-headstand,walker-flip,walker-backflip
# CHEETAH=cheetah-run,cheetah-run-backwards,cheetah-stand-front,cheetah-stand-back,cheetah-jump,cheetah-run-front,cheetah-run-back,cheetah-lie-down,cheetah-legs-up
# MANIPULATION=finger-spin,finger-turn-easy,finger-turn-hard,cup-catch,reacher-easy,reacher-hard
# CARTPOLE=cartpole-swingup,cartpole-balance,pendulum-swingup

# WALKERMINI=walker-walk,walker-run,walker-stand,walker-walk-backwards,walker-legs-up,walker-arabesque

# Checkpoint experiments
# python train_offline.py -m task=walker-walk modality=features features=mocodmcontrol exp_name=mocodmcontrol-chpkt hydra/launcher=slurm
# python train_offline.py -m task=walker-walk modality=features features=moco exp_name=moco-chpkt hydra/launcher=slurm
# python train_offline.py -m task=walker-walk modality=features features=mocoego15 exp_name=mocoego15-chpkt hydra/launcher=slurm
# python train_offline.py -m task=walker-walk modality=features features=mocoego50 exp_name=mocoego50-chpkt hydra/launcher=slurm
# python train_offline.py -m task=walker-walk modality=features features=mocoegodmcontrol exp_name=mocoegodmcontrol-chpkt hydra/launcher=slurm
# python train_offline.py -m task=walker-walk modality=features features=mocoego8crop exp_name=mocoego8crop-chpkt hydra/launcher=slurm
# python train_offline.py -m task=$WALKER modality=features features=mocoego15 exp_name=mocoego15-chpkt-l128 latent_dim=128 hydra/launcher=slurm
# python train_offline.py -m task=$WALKER modality=features features=mocoego15 exp_name=mocoego15-chpkt-l256 latent_dim=256 hydra/launcher=slurm


python train_offline.py \
    -m task=cup-catch,finger-spin,cheetah-run,walker-run,quadruped-run \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    exp_name=offline-v1 \
    seed=1,2,3 \
    hydra/launcher=slurm
