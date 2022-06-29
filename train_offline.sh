# Full task sets
WALKER=walker-walk,walker-run,walker-stand,walker-walk-backwards,walker-run-backwards,walker-arabesque,walker-lie-down,walker-legs-up,walker-headstand,walker-flip,walker-backflip
CHEETAH=cheetah-run,cheetah-run-backwards,cheetah-stand-front,cheetah-stand-back,cheetah-jump,cheetah-run-front,cheetah-run-back,cheetah-lie-down,cheetah-legs-up

WALKERMINI=walker-walk,walker-run,walker-arabesque,walker-flip,walker-backflip,walker-run-backwards,walker-headstand

# MoCo (DMControl) features
# python train_offline.py -m task=$WALKERMINI modality=features features=mocodmcontrol exp_name=mocodmcontrol-50eps enc_dim=512 mlp_dim=1024 hydra/launcher=slurm

# python train_offline.py -m task=$WALKERMINI modality=features features=mocodmcontrol +fuse=prep exp_name=mocodmcontrol-prepflare enc_dim=512 mlp_dim=1024 hydra/launcher=slurm
python train_offline.py task=walker-walk modality=state exp_name=test

# python train_offline.py algorithm=bc task=walker-walk modality=features features=mocodmcontrol exp_name=mocodmcontrol enc_dim=512 mlp_dim=1024 hydra/launcher=slurm

# Moco (ImageNet) features
# python train_offline.py -m task=$WALKERMINI modality=features features=moco exp_name=moco-again enc_dim=512 mlp_dim=1024 hydra/launcher=slurm

# MoCo (DMControl mini) features
# python train_offline.py task=walker-walk fraction=0.01 modality=features features=mocodmcontrolmini frame_stack=1 exp_name=mocodmcontrolmini enc_dim=512 mlp_dim=1024 hydra/launcher=slurm

# MoCo (DMControl-5m) features
# python train_offline.py -m task=$WALKER modality=features features=mocodmcontrol5m frame_stack=1 exp_name=mocodmcontrol5m enc_dim=512 mlp_dim=1024 hydra/launcher=slurm

# MoCo (Ego4D) features
# python train_offline.py -m task=$WALKERMINI modality=features features=mocoego50 exp_name=mocoego50-50eps enc_dim=512 mlp_dim=1024 hydra/launcher=slurm

# Encode dataset
# python encode_dataset.py -m task=$WALKER modality=pixels +features=mocoego15center hydra/launcher=slurm
