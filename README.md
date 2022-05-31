# Visual Representations for Control from Ego4D

This repo contains code for training visual representation models using Ego4D. See below for example usage.

## MoCo
- Install `mjrl` first and use the `pvr_beta_1` branch.
- Training MoCo requires a `.txt` file with absolute paths to all relevent frames. A data loader will be created to iterate over frames in the `.txt` file.
- Create the `.txt` with Ego4D frame paths using the file: `ego4d_ssl/moco/datasets/make_ego4d_list.py`
- Launch the training by following the example commands in `ego4d_ssl/moco/launch_job.sh`
- Change the logging config file to map to your checkpoints and wandb. See the file: `ego4d_ssl/moco/configs/moco/logging/default.yaml`
- For local training on devfair, a starter command is:
```
$ cd moco
$ PYTHONPATH=. python main_moco.py environment.slurm=False logging.name=local_exp data.train_filelist=datasets/ego4d_tiny.txt environment.ngpu=2 optim.epochs=20
```
- To launch experiment on FAIR cluster using slurm:
```
$ cd moco
$ PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=slurm_exp data.train_filelist=datasets/ego4d_tiny.txt environment.ngpu=8 environment.world_size=2 optim.epochs=2
```