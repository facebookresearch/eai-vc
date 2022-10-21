# Working with RSC


## Current limitations

* AirStore dependencies do not work with Python 3.9 so I changed the install instructions to create a conda environment with Python 3.8


## Cheat codes for RSC

The following notes allow you to be autonomous in deploying an environment on RSC. But there is a faster way to get your conda environment working on RSC. Just clone an existing environment!

Start by logging to RSC:

```bash
rsc -t
```
Then enable conda:

```bash
cd ~
module load anaconda3/2021.05
module load cuda/11.2
```

Then clone an existing environment:

```bash
conda create --name ov --clone /home/qduval/.conda/envs/ov
```

Then you can activate this environment:

```bash
source ~/.conda/envs/ov/bin/activate
```


## Setup your devserver

Please follow the instructions of this doc up to "Setup RSC Tool" (included):
https://docs.google.com/document/d/1vQRg_nyZrikerciidfsNQgRr-inkOhZPlGdgWqe37NE/edit#

Once done, please run the following:

```bash
source ~/.bash_profile
source ~/.bashrc
conda activate ov
```

That will take care that your conda environment and aliases are loaded in your command line.

### Create conda environment (DevServer)

Don't forget the `with-proxy` and also the Python version 3.8 and **not** 3.9:

```bash
with-proxy conda create --name ov python=3.8
conda activate ov
with-proxy conda install pytorch=1.12 torchvision=0.13 cudatoolkit=11.3 torchtext -c pytorch
```

### Install airstore

Make sure your are in the conda environment `ov` then run:

```bash
rsc_launcher install-airstore
```

_Note: This step will fail if you have python 3.9_

### Clone the omnivision repository

Make sure you already installed GIT on your devserver. If not follow these instructions:
https://www.internalfb.com/intern/wiki/Open_Source/Maintain_a_FB_OSS_Project/Devserver_GitHub_Access/#quick-setup

Then run the following command to clone the repository:

```bash
git clone git@github.com:fairinternal/omnivision.git
```

Make sure you use the SSH clone instructions (not the https ones) or else it will fail.

### Install the remaining dependencies

Install the dependencies of “omnivision” and the projects inside it:

```bash
cd omnivision
with-proxy pip install -e ".[dev,omniscale]"
with-proxy pip install -e projects
```

Don't forget the `with-proxy` or it will hang.

## Setup on RSC

Now it's time to deploy our code and conda environment on RSC.

### Sync your code and environment

Sync your code to RSC cluster

```bash
rsc rsync
```

Install conda-pack to send your conda environment to RSC

```bash
with-proxy conda install conda-pack
```

Package your environment:

```bash
cd ~/rsc
conda pack --ignore-editable-packages --name ov
```

This will create a tar file named `ov.tar.gz` in the `~/rsc` folder.

Then send this environment archive to RSC via:

```bash
rsc sync
```

### Deploy the environment

Log on to RSC using the following command:

```bash
rsc -t
```

Enable the necessary modules:

```bash
cd ~
module load anaconda3/2021.05
module load cuda/11.2
```

Then create the conda environment by first untar'ing the archive you created on your devserver:

```bash
mkdir -p .conda/envs/ov
tar -xzf ~/rsc/ov.tar.gz -C ~/.conda/envs/ov
```

And then we can activate it:

```bash
source ~/.conda/envs/ov/bin/activate
conda-unpack
```


## Did my install work?

### Run the unit tests

You can run a unit tests on RSC to check that omnivision is installed correctly.

First log to RSC if not done already:

```bash
rsc -t
```

Activate your conda environment:

```bash
source ~/.conda/envs/ov/bin/activate
```

Then run the tests:

```bash
cd ~/rsc/omnivision/tests
python -m unittest test_model_wrappers.py
```

### Run a dummy job on SLURM

As a second check, you can run a dummy job to see if you can schedule a job on RSC.

First log to RSC if not done already:

```bash
rsc -t
```

Activate your conda environment:

```bash
source ~/.conda/envs/ov/bin/activate
```

Go under the project folder:

```bash
cd ~/rsc/omnivision/projects/omnivore
```

Then you can start training with the `dev/launch_job.py` command as usual:

```bash
./dev/launch_job.py -c config/experiments/reference/dummy_kinetics_train_slurm_gpu_rsc.yaml
```

This should schedule a job and you should see the logs in `$HOME/log_runs`.


## After the install: working on Devserver + RSC

### Enable Conda environment

Switch to `ov` conda environment on devserver:

```bash
source ~/.bash_profile
source ~/.bashrc
source ~/anaconda3/bin/activate ov
```

Switch to `ov` conda environment on RSC server:

```bash
source ~/.conda/envs/ov/bin/activate
```

### Add a new dependency

Once the RSC environment is set up, code will have to evolve and new dependencies might need to be added. This section contains the command needed to work at head version on RSC.

First install the new dependency in the `ov` environment on your devserver.

Then package the updated environment using `conda pack`:

```bash
cd ~/rsc
rm ov.tar.gz
conda pack --ignore-editable-packages --name ov
```

Then sync this updated version to RSC (the time it takes depends on the size of the conda environment):

```bash
rsc sync
```

Finally, replace the environment in RSC with the updated one:

```bash
tar -xzf ~/rsc/ov.tar.gz -C ~/.conda/envs/ov; conda-unpack
```

### Update code following edits

Just sync with the RSC:

```bash
rsc sync
```

Note however that it might take time due to the archive used to package the conda environment. You can speed up the sync of the code by focusing on the code folder only:

```bash
rsc sync omnivision
```


### Launching Jobs with Aistore dataloader,
After setting up the RSC using the above instructions, from your devserver you can launch Airstore jobs using the command,
```
rsc sync && rsc_launcher launch -p <YOUR TABLE SPECIFIC CRYPTO KEY>  -e "module load anaconda3/2021.05; module load cuda/11.2; source ~/.conda/envs/ov/bin/activate; cd /home/$USER/rsc/omnivision/projects/omnivore; ./dev/launch_job.py --partition learn --force -c config/experiments/<PATH TO YOUR CONFIGS>.yaml"
```

For instance, to launch MAE training on the Uru 10x10 dataset,
```
rsc sync && rsc_launcher launch -p AIRSTORE_OMNISCALE_URU10X10_CAPTION_CRYPTO  -e "module load anaconda3/2021.05; module load cuda/11.2; source ~/.conda/envs/ov_rsc/bin/activate; cd /home/mannatsingh/rsc/omnivision/projects/omnivore; ./dev/launch_job.py -c config/experiments/mannatsingh/mae/pretrain/vit_h_uru5b_1_epoch_bs_4k_gpus_128.yaml"
```
