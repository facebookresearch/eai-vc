python hydra_launcher.py --config-name DMC_BC_config.yaml --multirun hydra/launcher=slurm \
env=dmc_walker_stand-v1,dmc_walker_walk-v1,dmc_reacher_easy-v1,dmc_cheetah_run-v1,dmc_finger_spin-v1 \
embedding=$(python -m eaif_models)