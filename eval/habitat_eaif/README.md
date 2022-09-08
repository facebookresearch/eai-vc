## Usage

The repository can be used to train ImageNav and ObjectNav agents in the Habitat simulator

### ImageNav
**Note:** If you are on the FAIR cluster, run the following command to symlink the habitat related dataset folder into your local directory:
`./eval/habitat_eaif/data/symlink.sh`

Otherwise, follow the instructions on [habitat-lab](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md) to get the Gibson scene and Gibson ImageNav episode datasets.

1. To run experiments in Habitat, first we need to get access to the necessary scene dataset. We are using Gibson scene datasets for our ImageNav experiment. You can find instructions for downloading the dataset [here](https://github.com/facebookresearch/habitat-lab#gibson).

1. Next we need the episode dataset for ImageNav. You can get the training and validation dataset from here [TODO].
   
1. Finally, download the pretrained models and add them to the folder `data/ddppo-models`. You can download the Dino ResNet50 model from [here](https://www.dropbox.com/s/xd5fsuqamen6ov7/omnidata_DINO_02.pth?dl=0) and MAE ViT-Small model from [here](https://www.dropbox.com/s/jndr74khagi3ndc/mae_vit_small_decoder_large_HGPS_RE10K_100.pth?dl=0).

1. Now we are ready to start training the agent. Checkout the `example_run.sh` script, which allows running an experiment on the cluster. The script can be used as follows:
   ```
   ./sbatch_scripts/example_run.sh scratch train
   ```
   Change `scratch` to `dino`, `mae` or `d2v` to train an agent with one of those pretrained models.

1. Once you have trained a model, it is time for evaluation. We evaluate every 5th saved checkpoint. To run an evaluation, just change the mode from `train` to `eval`.
   ```
   ./sbatch_scripts/example_run.sh scratch eval
   ```
