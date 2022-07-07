# xvfb-run -a python train_offline.py task=rlb-close-drawer modality=pixels frame_stack=1 action_repeat=1 episode_length=125 exp_name=test
python train.py \
    -m task=rlb-reach-target \
    modality=state \
    enc_dim=256 \
    mlp_dim=512 \
    per=true \
    action_repeat=2 \
    episode_length=100 \
    train_steps=500000 \
    eval_freq=50000 \
    save_freq=50000 \
    exp_name=v1 \
    seed=1 \
    hydra/launcher=slurm
