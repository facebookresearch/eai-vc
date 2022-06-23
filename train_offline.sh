# Full task sets
WALKER=walker-walk,walker-run,walker-stand,walker-walk-backwards,walker-run-backwards,walker-arabesque,walker-lie-down,walker-legs-up,walker-headstand,walker-flip,walker-backflip
CHEETAH=cheetah-run,cheetah-run-backwards,cheetah-stand-front,cheetah-stand-back,cheetah-jump,cheetah-run-front,cheetah-run-back,cheetah-lie-down,cheetah-legs-up

WALKERMINI=walker-walk,walker-run,walker-arabesque,walker-flip,walker-backflip,walker-run-backwards


# MoCo (DMControl) features
python train_offline.py -m task=$WALKERMINI modality=features features=mocodmcontrol exp_name=mocodmcontrol-flare-L-b4096 batch_size=4096 enc_dim=512 mlp_dim=1024 hydra/launcher=slurm
