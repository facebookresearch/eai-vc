python train.py \
    task=walker-walk \
    suite=dmcontrol \
    setting=online \
    modality=map \
    features=mocoego18 \
    feature_dims=[32,14,14] \
    +distractors=true \
    exp_name=test \
    seed=1
