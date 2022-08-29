python hydra_launcher.py --config-name DMC_BC_config.yaml --multirun hydra/launcher=slurm \
env=dmc_walker_stand-v1,dmc_walker_walk-v1,dmc_reacher_easy-v1,dmc_cheetah_run-v1,dmc_finger_spin-v1 \
embedding=dino_omnidata,mae_large,mae_large_ego4d,mae_small_HGSP_RE10K_100,mae_small_ego4d,moco,moco_ego4d,r3m,rn50_rand,rn50_sup_imnet #wandb=aravraj wandb.project=eaif-mujoco-test
