CUDA_VISIBLE_DEVICES=0 python train.py \
    task=mw-push \
    suite=mw \
    setting=online \
    modality=map \
    features=mocoego18 \
    exp_name=test \
    seed=1
