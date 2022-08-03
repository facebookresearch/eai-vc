## Installation

1. Create a conda environment:
   ```
   conda env create -f environment.yml

   conda activate eai
   ```

1. Install mae-for-eai:

   ```
   pip install -e .
   ```

1. Install [habitat-lab](https://github.com/facebookresearch/habitat-lab/tree/v0.2.1) version `0.2.1`:
   ```
   cd third_party/habitat-lab

   # patch requirements file
   head -2 habitat_baselines/rl/requirements.txt > tmp && mv tmp habitat_baselines/rl/requirements.txt

   pip install -r requirements.txt

   python setup.py develop --all # install habitat and habitat_baselines
   ```

1. Follow the instructions [here](https://github.com/facebookresearch/habitat-lab#data) to set up the `data/scene_datasets/` directory. Your directory structure should now look like this:
   ```
   +-- mae-for-eai/
   |   +-- habitat-lab-v0.2.1/
   |   +-- data/
   |   |   +-- datasets/
   |   |   |   +-- imagenav/
   |   |   +-- scene_datasets/
   |   |   |   +-- gibson/
   |   +-- eai/
   |   ...
   ```

## Usage

The repository can be used to train:

   1. Visual encoders using MAE (`mae` folder)
   1. Visual encoders using TMAE (`tmae` folder)
   1. ImageNav and ObjectNav agents in the Habitat simulator  (`eai` folder)

### MAE
To train a visual encoder using MAE, run the following commands:
1. Change directory to mae:
   ```
   cd mae/
   ```

1. Run the following command in the terminal. Make sure to change the `--data_path` and `--output_dir` paths. If you are on the FAIR cluster, you can also use my `data_path`. 
   ```
   python submitit_pretrain.py \
      --wandb_name mae_vit_small_HGSP \
      --nodes 2 \
      --batch_size 256 \
      --model mae_vit_base_patch16 \
      --norm_pix_loss \
      --mask_ratio 0.75 \
      --epochs 400 \
      --warmup_epochs 40 \
      --blr 1.5e-4 --weight_decay 0.05 \
      --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train \
      --output_dir /checkpoint/karmeshyadav/mae_training/ \
      --partition learnlab --use_volta32
   ```

### TMAE
To train a visual encoder using MAE, run the following commands:
1. Change directory to tmae:
   ```
   cd tmae/
   ```

1. Run the following command in the terminal. Make sure to change the `--data_path` and `--output_dir` paths. If you are on the FAIR cluster, you can also use my `data_path`.
   ```
   python submitit_pretrain.py \
      --wandb_name tmae_small_offset_4_HGSP \
      --nodes 4 \
      --batch_size 128 \
      --accum_iter 1 \
      --model mae_vit_small_patch16 \
      --norm_pix_loss \
      --max_offset 4 \
      --mask_ratio1 0.75 \
      --mask_ratio2 0.95 \
      --loss_weight 0.5 \
      --epochs 400 \
      --warmup_epochs 40 \
      --randomize_views \
      --blr 1.5e-4 --weight_decay 0.05 \
      --data_path /checkpoint/karmeshyadav/hm3d+gibson/v1/train/ \
      --output_dir /checkpoint/karmeshyadav/mae_training/ \
      --partition learnlab --use_volta32
   ```

### EAI
#### ImageNav

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
