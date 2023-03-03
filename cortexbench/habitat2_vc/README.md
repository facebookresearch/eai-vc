## Habitat 2.0: MobilePick

Habitat 2.0 (Szot et al., 2021) ) is a simulation platform for training virtual robots in interactive 3D environments and complex physics-enabled scenarios. It includes a set of mobile manipulation tasks in which an agent controls a Fetch robot with a 7-DoF arm, mobile base and suction gripper to rearrange objects in apartment scenes. 

We consider a challenging version of the Mobile-Pick task from Habitat 2.0, in which an agent must pick up a target object from a cluttered receptacle (e.g., a counter) while starting from a position in which the object is outside of the robot’s reach (thus, requiring navigation) without counting wih a dense goal specification sensor.


## Setup 

This benchmark requires a different environment because it uses a different version of habitat (0.2.3). To 
install the environment see the [installation instructions](INSTALLATION.md).

Also make sure to specify your [output folder](configs/hydra/output/path.yaml) for local runs and sweeps as well as your [wandb entity](configs/wandb_habitat/habitat2.yaml) to log results.

## Run

Run the following command to start training locally:

```
python run_rearrangement_vc.py model=vc1_vitb WANDB.run_name=<run_name> habitat_baselines.num_environments=6
```
> Model can be vc1_vitb and vc1_vitl

You can use hydra to submit the job to a cluster by running:

```
python run_rearrangement_vc.py model=vc1_vitb WANDB.run_name=<run_name> habitat_baselines.num_environments=6 hydra/launcher=slurm_train_base -m
```

To evaluate a model run:

```
python run_rearrangement_vc.py model=vc1_vitb WANDB.run_name=<same_run_name_as_train> habitat_baselines.num_environments=10 hydra/launcher=slurm_eval_base RUN_TYPE=eval -m
```


