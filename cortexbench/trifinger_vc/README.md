# Trifinger Behavior Cloning Benchmark

## Installation
This package can be installed using pip ('pip install -e . from `cortexbench/trifinger_vc`). To run this package, trifinger_simulation must also be installed `./third_party/trifinger_simulation`. 


## Overview
The trifinger package includes the environments, controls for the robot, and training code and related configuration files for running behavior cloning.

The downstream task for this involves moving a cube using the Trifinger robot from a sampled start position to an image-specified goal position. In the MoveCubeEnv, the action is specified as the fingertip displacements. In addition to this, there is an environment, with the trifinger robot and a cube on the table, where the objective is for the agent to reach one finger as close to the center of the cube as possible.

The policies are learned from behavior cloning. The demonstrations are available to be downloaded at [this link](https://dl.fbaipublicfiles.com/eai-vc/trifinger_demos.zip). When running the training script, the files are automatically downloaded from that link to 'cortexbench/trifinger_vc/assets/data/trifinger-demos/'. There are 125 demonstrations for each the reach and the move task. The json files in 'assets/bc_demos/' list out the train/test splits for the datasets.
Within each demo folder, there exists a file `downsample.pth` and a gif of the dmeonstration. 

The demo can be loaded using torch.load(filename) and will contain a dictionary with the following keys: 't', 'o_pos_cur', 'o_pos_des', 'o_ori_cur', 'o_ori_des', 'ft_pos_cur', 'ft_pos_des', 'ft_vel_cur', 'ft_vel_des', 'position_error', 'robot_pos', 'image_60', 'image_180', 'image_300', 'ft_pos_targets_per_mode', 'mode', 'vertices', 'delta_ftpos'. 

The o_pos refer to the object position (in our case the cube) and ft_pos refers to the fingertip position of the trifinger robot. At each recorded time step(t), both the desired and current position of the cube and fingertips are tracked. Along with that, there are images from various angles, the robot's joint positions, and the change in fingertips from the previus step. 



## Training
First make sure you have the environment set up:
```bash
conda activate eai-vc
pip install -e ./vc_models #if not already done
pip install -e ./third_party/trifinger_simulation #if not already done
pip install -e ./cortexbench/trifinger_vc #if not already done
cd cortexbench/trifinger_vc
# To begin training using demonstrations, call
python bc_experiments/train_bc.py seed=<SEED> algo.pretrained_rep=vc1_vitb algo.freeze_pretrained_rep=true rep_to_policy=none task=reach_cube task.n_outer_iter=<NUM_ITR> no_wandb=True run_name=<WANDB_RUN_NAME>```

