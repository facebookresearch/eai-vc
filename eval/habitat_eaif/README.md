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
   python run_habitat_eaif.py -m 
   ```
   This will start a run on the slurm with the default folder name `imagenav_run`.

1. If you want to start a local run, add `hydra/launcher=slurm` at the end of the command listed in the previous point.

1. Once you have trained a model, it is time for evaluation. We evaluate every 5th saved checkpoint. To run an evaluation, just change the mode from `train` to `eval`.
   ```
   python run_habitat_eaif.py -m RUN_TYPE=eval hydra/launcher=slurm_eval NUM_ENVIRONMENTS=14
   ```
