# Set W&B to public instance to log to shared team
export WANDB_BASE_URL="https://api.wandb.ai"

# DMC
python hydra_launcher.py --config-name DMC_BC_config.yaml --multirun hydra/launcher=slurm \
        wandb.project=dmc_test wandb.entity=eai-foundations \
        env=dmc_walker_stand-v1,dmc_walker_walk-v1,dmc_reacher_easy-v1,dmc_cheetah_run-v1,dmc_finger_spin-v1 \
           embedding=$(python -m core_model_set)

# Adroit
python hydra_launcher.py --config-name Adroit_BC_config.yaml --multirun hydra/launcher=slurm \
    wandb.project=adroit_test wandb.entity=eai-foundations \
    env=pen-v0,relocate-v0 embedding=$(python -m core_model_set)

# Metaworld
python hydra_launcher.py --config-name Metaworld_BC_config.yaml --multirun hydra/launcher=slurm \
        wandb.project=metaworld_test wandb.entity=eai-foundations \
        env=assembly-v2-goal-observable,bin-picking-v2-goal-observable,button-press-topdown-v2-goal-observable,drawer-open-v2-goal-observable,hammer-v2-goal-observable \
        embedding=$(python -m core_model_set)
