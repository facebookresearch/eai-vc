python train_offline.py -m \
    task=mw-mt10 \
    suite=mw \
    setting=multitask \
    algorithm=mtdmpc \
    modality=features \
    features=mocoego \
    latent_dim=50 \
    mlp_dim=512 \
    exp_name=offline-v2-prop-fast-small \
    seed=1,2,3 \
    hydra/launcher=slurm \
    hydra.job.name=mtdmpc-features \
    hydra.launcher.timeout_min=2900 \
    hydra.launcher.mem_gb=96 \
    hydra.launcher.cpus_per_task=20
