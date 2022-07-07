# Full task sets
WALKER=walker-walk,walker-run,walker-stand,walker-walk-backwards,walker-run-backwards,walker-arabesque,walker-lie-down,walker-legs-up,walker-headstand,walker-flip,walker-backflip
CHEETAH=cheetah-run,cheetah-run-backwards,cheetah-stand-front,cheetah-stand-back,cheetah-jump,cheetah-run-front,cheetah-run-back,cheetah-lie-down,cheetah-legs-up

WALKERMINI=walker-walk,walker-run,walker-stand,walker-walk-backwards,walker-legs-up,walker-arabesque

MANIPULATION=finger-spin,finger-turn-easy,finger-turn-hard,cup-catch,reacher-easy,reacher-hard
CARTPOLE=cartpole-swingup,cartpole-balance,pendulum-swingup

# Encode dataset
python encode_dataset.py \
    -m task=cup-catch,finger-spin,walker-run \
    modality=pixels \
    +features=random \
    hydra/launcher=slurm
