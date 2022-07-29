python train_offline.py \
    task=mw-hammer \
    suite=mw \
    setting=offline \
    algorithm=tdmpc \
    modality=pixels \
    +include_state=true \
    fraction=0.01 \
    exp_name=test \
    seed=1
