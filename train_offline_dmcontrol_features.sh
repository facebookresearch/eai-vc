python train_offline.py \
    -m task=cup-catch,finger-spin,cheetah-run,walker-run,quadruped-run \
    modality=features \
    features=mocoego,random \
    exp_name=v1 \
    seed=1,2,3 \
    hydra/launcher=slurm
