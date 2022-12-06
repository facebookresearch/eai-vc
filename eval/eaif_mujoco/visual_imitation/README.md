## Visual Imitation Learning with Frozen PVRs

To launch a test experiment on each of the suites (dmc, adroit, metaworld), you can use the following commands

### DM Control

```
python hydra_launcher.py --config-name DMC_BC_config.yaml --multirun hydra/launcher=slurm \
	env=dmc_walker_stand-v1 embedding=r3m_resnet50_ego4d \
	wandb.project=cortex_dmc_test wandb.entity=aravraj
```

### Adroit
The below runs will add proprioception to policy input by default

```
python hydra_launcher.py --config-name Adroit_BC_config.yaml --multirun hydra/launcher=slurm \
        env=pen-v0 embedding=r3m_resnet50_ego4d \
        wandb.project=cortex_adroit_test wandb.entity=aravraj

python hydra_launcher.py --config-name Adroit_BC_config.yaml --multirun hydra/launcher=slurm \
        env=relocate-v0 embedding=r3m_resnet50_ego4d \
        wandb.project=cortex_adroit_test wandb.entity=aravraj
```

### MetaWorld
The below runs will add proprioception to policy input by default

```
python hydra_launcher.py --config-name Metaworld_BC_config.yaml --multirun hydra/launcher=slurm \
        env=drawer-open-v2-goal-observable embedding=r3m_resnet50_ego4d \
        wandb.project=cortex_metaworld_test wandb.entity=aravraj
```

