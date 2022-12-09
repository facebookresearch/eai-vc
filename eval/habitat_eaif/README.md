## Usage

The repository can be used to train ImageNav and ObjectNav agents in the Habitat simulator

### ImageNav
**Note:** If you are on the FAIR cluster, run the following command to symlink the habitat related dataset folder into your local directory:
`./eval/habitat_eaif/data/symlink.sh`

Otherwise, follow the instructions on [habitat-lab](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md) to get the Gibson scene and Gibson ImageNav episode datasets.

1. To run experiments in Habitat, first we need to get access to the necessary scene dataset. We are using Gibson scene datasets for our ImageNav experiment. You can find instructions for downloading the dataset [here](https://github.com/facebookresearch/habitat-lab#gibson).

1. Next we need the episode dataset for ImageNav. You can get the training and validation dataset from here [TODO].

1. Now we are ready to start training the agent. Checkout the `run_habitat_eaif.py` script, which allows running an experiment on the cluster. The script can be used as follows:
   ```
   python run_habitat_eaif.py --config-name=config_imagenav -m 
   ```
   This will start a run on the slurm with the default folder name `imagenav_run`.

1. If you want to start a local run, add `hydra/launcher=slurm` at the end of the command listed in the previous point.

1. Once you have trained a model, it is time for evaluation. We evaluate every 5th saved checkpoint. To run an evaluation, do the following:
   ```
   python run_habitat_eaif.py --config-name=config_imagenav RUN_TYPE=eval hydra/launcher=slurm_eval NUM_ENVIRONMENTS=14 -m
   ```

### ObjectNav
1. For our ObjectNav RL experiments we will be using the HM3DSem v0.2 scene dataset and the corresponding HM3DSem v0.2 ObjectNav episode dataset. Currently the data is available in `/checkpoint/yixinlin/eaif/datasets/habitat_task_dataset/datasets/objectnav/hm3d/v0.2/` on the FAIR cluster. The data is automatically added to the `habitat_eaif` folder when you run the `symlink.sh` file.

1. To start a training run with the `dino_resnet50_omnidata` model use the following command: 
   ```
   python run_habitat_eaif.py --config-name=config_objectnav_rl WANDB.name=Objectnav_first_experiment model=dino_resnet50_omnidata -m
   ```

1. Once you have trained a model, it is time for evaluation. We evaluate every 5th saved checkpoint. To run an evaluation, do the following:
   ```
   python run_habitat_eaif.py --config-name=config_objectnav_rl WANDB.name=Objectnav_first_experiment model=dino_resnet50_omnidata RUN_TYPE=eval hydra/launcher=slurm_eval NUM_ENVIRONMENTS=30 -m
   ```

### ObjectNav
1. For our ObjectNav RL experiments we will be using the HM3DSem v0.2 scene dataset and the corresponding HM3DSem v0.2 ObjectNav episode dataset. Currently the data is available in `/checkpoint/yixinlin/eaif/datasets/habitat_task_dataset/datasets/objectnav/hm3d/v0.2/` on the FAIR cluster. The data is automatically added to the `habitat_eaif` folder when you run the `symlink.sh` file.

1. To start a training run with the `dino_resnet50_omnidata` model use the following command: 
   ```
   python run_habitat_eaif.py --config-name=config_objectnav_rl WANDB.name=Objectnav_first_experiment model=dino_resnet50_omnidata -m
   ```

1. Once you have trained a model, it is time for evaluation. We evaluate every 5th saved checkpoint. To run an evaluation, do the following:
   ```
   python run_habitat_eaif.py --config-name=config_objectnav_rl WANDB.name=Objectnav_first_experiment model=dino_resnet50_omnidata RUN_TYPE=eval hydra/launcher=slurm_eval NUM_ENVIRONMENTS=20
   ```
