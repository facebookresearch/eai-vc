## Usage

The repository can be used to train ImageNav and ObjectNav agents in the Habitat simulator

### ImageNav
**Note:** If you are on the FAIR cluster, the symlinks should already point to Karmesh's dataset folder `/private/home/karmeshyadav/mae/mae-for-eai/data/`. Otherwise, follow the instructions at each step to get the necessary datasets. [TODO: put them in a more permanent directory]

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
