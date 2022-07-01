# Full task sets
WALKER=walker-walk,walker-run,walker-stand,walker-walk-backwards,walker-run-backwards,walker-arabesque,walker-lie-down,walker-legs-up,walker-headstand,walker-flip,walker-backflip
CHEETAH=cheetah-run,cheetah-run-backwards,cheetah-stand-front,cheetah-stand-back,cheetah-jump,cheetah-run-front,cheetah-run-back,cheetah-lie-down,cheetah-legs-up

WALKERMINI=walker-walk,walker-run,walker-stand,walker-walk-backwards,walker-legs-up,walker-arabesque

# Checkpoint experiments
# python train_offline.py -m task=walker-walk modality=features features=mocodmcontrol exp_name=mocodmcontrol-chpkt hydra/launcher=slurm
# python train_offline.py -m task=walker-walk modality=features features=moco exp_name=moco-chpkt hydra/launcher=slurm
# python train_offline.py -m task=walker-walk modality=features features=mocoego15 exp_name=mocoego15-chpkt hydra/launcher=slurm
# python train_offline.py -m task=walker-walk modality=features features=mocoego50 exp_name=mocoego50-chpkt hydra/launcher=slurm
# python train_offline.py -m task=walker-walk modality=features features=mocoegodmcontrol exp_name=mocoegodmcontrol-chpkt hydra/launcher=slurm
# python train_offline.py -m task=walker-walk modality=features features=mocoego8crop exp_name=mocoego8crop-chpkt hydra/launcher=slurm
# python train_offline.py -m task=$WALKER modality=features features=mocoego15 exp_name=mocoego15-chpkt-l128 latent_dim=128 hydra/launcher=slurm
# python train_offline.py -m task=$WALKER modality=features features=mocoego15 exp_name=mocoego15-chpkt-l256 latent_dim=256 hydra/launcher=slurm


python train_offline.py -m task=$WALKERMINI modality=features features=mocoego15 dynamics_obj=reconstruction consistency_coef=100 detach_rewval=true exp_name=mocoego15-recon-c100-detach hydra/launcher=slurm



# Encode dataset
# python encode_dataset.py -m task=$WALKER modality=pixels +features=mocoego8crop hydra/launcher=slurm
