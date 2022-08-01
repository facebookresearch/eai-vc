CUDA_VISIBLE_DEVICES=0 python train.py \
    task=walker-walk \
    suite=dmcontrol \
    setting=online \
    modality=map \
    features=mocoego18 \
    exp_name=test \
    seed=1
