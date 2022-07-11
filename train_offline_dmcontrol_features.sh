python train_offline.py \
    -m task=cup-catch,finger-spin,cheetah-run,walker-run,quadruped-run \
    modality=features \
    features=mocodmcontrol \
    exp_name=offline-v1 \
    seed=1,2,3 \
    hydra/launcher=slurm
