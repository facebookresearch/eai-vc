# Habitat: ImageNav and ObjectNav

### ImageNav
1. To run experiments in Habitat, first we need to get access to the necessary scene dataset. We are using Gibson scene datasets for our ImageNav experiment. You can find instructions for downloading the dataset [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets).

1. Next we need the episode dataset for ImageNav. You can get the training and validation dataset from [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) and place it in the [./data](./data) folder under the path : `data/datasets/pointnav/gibson/v1/`. 

1. Now we are ready to start training the agent. Checkout the `run_habitat_vc.py` script, which allows running an experiment on the cluster. The script can be used as follows:
   ```bash
   python run_habitat_vc.py --config-name=config_imagenav -m 
   ```
   This will start a run on the slurm with the default folder name `imagenav_run`.

1. If you want to start a local run, add `hydra/launcher=slurm` at the end of the command listed in the previous point.

1. Once you have trained a model, it is time for evaluation. We evaluate every 5th saved checkpoint. To run an evaluation, do the following:
   ```bash
   python run_habitat_vc.py --config-name=eval_config_imagenav hydra/launcher=slurm_eval NUM_ENVIRONMENTS=14 -m
   ```

### ObjectNav
1. For our ObjectNav IL experiments we will be using the HM3DSem v0.1 scene dataset and the corresponding HM3DSem v0.1 ObjectNav episode dataset. Currently the data is available by [following these instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#downloading-hm3d-with-the-download-utility). Download the Scene dataset `HM3D` 

1. The Task training dataset linked [here](https://habitat-on-web.s3.amazonaws.com/pirlnav_release/objectnav_hm3d_hd.zip) which contents you need to place in `data/datasets/objectnav/hm3d/objectnav_hm3d_77k/`

1. The Task validation dataset linked [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip) which contents you need to place in `data/datasets/objectnav/hm3d/v1/`

1. To start a training run with the `vc1_vitl` model use the following command: 
   ```bash
      python run_habitat_vc.py --config-name=config_objectnav_il_frozen WANDB.name=Objectnav_first_experiment model=vc1_vitl -m
   ```

1. Once you have trained a model, it is time for evaluation. We evaluate every 5th saved checkpoint. To run an evaluation, do the following:
   ```bash
   python run_habitat_vc.py --config-name=eval_config_objectnav_il_frozen WANDB.name=Objectnav_first_experiment model=vc1_vitl  hydra/launcher=slurm_eval NUM_ENVIRONMENTS=30 -m
   ```