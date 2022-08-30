# Installation

Clone the repo:

```bash
git clone git@github.com:facebookresearch/eai-foundations.git
cd eai-foundations

git submodule update --init --recursive  # Also necessary if we updated any submodules
```

Create the Conda environment:

```bash
# Recommended: conda install -c conda-forge -n base mamba
# Replace all conda commands with mamba

conda env create -f environment.yml
conda activate eaif  # Alternatively, `direnv allow`
```

Some notes on the conda environment:
- If conda is unable to resolve the environment, you may need to update conda to a newer version (e.g. try downloading a local version of miniconda instead of the module-loaded cluster-wide one).
- [mamba](https://github.com/mamba-org/mamba) is a significantly faster drop-in replacement for conda.
- [direnv](https://direnv.net/) with the [anaconda](https://github.com/direnv/direnv/wiki/Python#anaconda) plugin removes the need to `conda activate` when in current working directory.

Setup Mujoco/mj_envs/mjrl:

```bash
# If you don't have Mujoco yet: copy Mujoco files, keys
cp -r /checkpoint/yixinlin/eaif/libraries/mujoco/. ~/.mujoco

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
pip install -e ./eval/trifinger/rl-utils
pip install -e ./eval/trifinger/imitation_learning
```

Install local packages:


```bash
pip install -e ./eaif-models  # Install model-loading API
pip install -e ./train/mae-for-eai  # Install Habitat tasks
pip install -e ./eval/eaif_mujoco  # Install Visual IL tasks
```

## Optional

Test Mujoco is using GPU to render:

```bash
cd ./third_party/mujoco-py
python ./examples/multigpu_rendering.py

# Expected output with GPU:
: '
Found 5 GPUs for rendering. Using device 0.
main(): start benchmarking                         
Found 5 GPUs for rendering. Using device 1.
Completed in 0.3s: 1.354ms, 738.7 FPS 
'

# Expected output without GPU:
: '
main(): start benchmarking                 
Completed in 2.7s: 13.451ms, 74.3 FPS
'
```

If you are unable to load `mujoco_py` with error `ImportError: cannot import name 'load_model_from_path' from 'mujoco_py' (unknown location)`, try running

```bash
rm -rf ~/.local/lib/python3.8/site-packages/mujoco_py
```
