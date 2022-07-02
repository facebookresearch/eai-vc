# Full task sets
WALKER=walker-walk,walker-run,walker-stand,walker-walk-backwards,walker-run-backwards,walker-arabesque,walker-lie-down,walker-legs-up,walker-headstand,walker-flip,walker-backflip
CHEETAH=cheetah-run,cheetah-run-backwards,cheetah-stand-front,cheetah-stand-back,cheetah-jump,cheetah-run-front,cheetah-run-back,cheetah-lie-down,cheetah-legs-up

WALKERMINI=walker-walk,walker-run,walker-stand,walker-walk-backwards,walker-legs-up,walker-arabesque

MANIPULATION=finger-spin,finger-turn-easy,finger-turn-hard,cup-catch,reacher-easy,reacher-hard

# Encode dataset
# python encode_dataset.py -m task=$WALKER modality=pixels +features=mocoego15center hydra/launcher=slurm
python encode_dataset.py -m task=$MANIPULATION modality=pixels +features=mocoego,mocodmcontrol hydra/launcher=slurm
