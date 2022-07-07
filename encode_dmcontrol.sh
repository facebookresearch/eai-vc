python encode_dataset.py \
    -m task=cup-catch,finger-spin,cheetah-run,walker-run,quadruped-run \
    modality=pixels \
    +features=mocoego,random \
    hydra/launcher=slurm
