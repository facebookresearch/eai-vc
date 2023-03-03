# Installation

Clone the repo:

```bash
git clone  https://github.com/facebookresearch/eai-vc.git
cd eai-vc

git submodule update --init --recursive  # Also necessary if we updated any submodules
```

[Install Conda package manager](https://docs.conda.io/en/latest/miniconda.html). Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate eai-vc  # Alternatively, `direnv allow`
```

Setup Mujoco/mj_envs/mjrl:
```bash
mkdir ~/.mujoco
# Go to https://www.roboti.us/download.html to download Mujoco library
wget https://www.roboti.us/download/mujoco200_linux.zip -P ~/.mujoco
unzip ~/.mujoco/mujoco200_linux.zip

# Go to https://www.roboti.us/license.html to obtain the key under their Free license:
wget https://www.roboti.us/file/mjkey.txt -P ~/.mujoco
```

```bash
# Install mujoco-py (GPU-compiled)
pip install -e ./third_party/mujoco-py

# Install mj_envs/mjrl
pip install -e ./third_party/mj_envs
pip install -e ./third_party/mjrl
pip install -e ./third_party/dmc2gym
```

Install Habitat-Lab v0.2.1 (patched to remove Tensorflow dependencies):

```bash
cd third_party/habitat-lab
python setup.py develop --all # install habitat and habitat_baselines
cd -
```

Install the Trifinger environment:

```bash
pip install -e ./third_party/trifinger_simulation
```

Install local packages:


```bash
pip install -e ./vc_models  # Install model-loading API
pip install -e ./cortexbench/mujoco_vc  # Install Visual IL tasks
pip install -e ./cortexbench/habitat_vc  # Install Habitat tasks
pip install -e ./cortexbench/trifinger_vc  # Install Habitat tasks
```

To use the Habitat 2.0 Rearrangement benchmark, you need to set up a separate conda environment. The steps for doing this are described in the installation instructions, which can be found at [cortexbench/habitat2_vc/INSTALLATION.md](cortexbench/habitat2_vc/INSTALLATION.md).

If you are unable to load `mujoco_py` with error `ImportError: cannot import name 'load_model_from_path' from 'mujoco_py' (unknown location)`, try running

```bash
rm -rf ~/.local/lib/python3.8/site-packages/mujoco_py
```

Furthermore, the recipe for running each benchmark on a bare Linux environment is included in our CircleCI job configuration, which can be found in the [`.circleci/config.yml`](.circleci/config.yml) file.