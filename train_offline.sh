# Full task sets
WALKER=walker-walk,walker-run,walker-stand,walker-walk-backwards,walker-run-backwards,walker-arabesque,walker-lie-down,walker-legs-up,walker-headstand,walker-flip,walker-backflip
CHEETAH=cheetah-run,cheetah-run-backwards,cheetah-stand-front,cheetah-stand-back,cheetah-jump,cheetah-run-front,cheetah-run-back,cheetah-lie-down,cheetah-legs-up

WALKERMINI=walker-walk,walker-run,walker-arabesque,walker-flip,walker-backflip,walker-run-backwards

# MoCo (DMControl) features
python train_offline.py -m task=$WALKERMINI modality=features features=mocodmcontrol exp_name=mocodmcontrol-again enc_dim=512 mlp_dim=1024 hydra/launcher=slurm

# Moco (ImageNet) features
# python train_offline.py -m task=$WALKERMINI modality=features features=moco exp_name=moco-again enc_dim=512 mlp_dim=1024 hydra/launcher=slurm

# MoCo (DMControl mini) features
# python train_offline.py task=walker-walk fraction=0.01 modality=features features=mocodmcontrolmini frame_stack=1 exp_name=mocodmcontrolmini enc_dim=512 mlp_dim=1024 hydra/launcher=slurm

# MoCo (DMControl-5m) features
# python train_offline.py -m task=$WALKER modality=features features=mocodmcontrol5m frame_stack=1 exp_name=mocodmcontrol5m enc_dim=512 mlp_dim=1024 hydra/launcher=slurm

# Encode dataset
# python encode_dataset.py -m task=$WALKER modality=pixels +features=moco hydra/launcher=slurm
