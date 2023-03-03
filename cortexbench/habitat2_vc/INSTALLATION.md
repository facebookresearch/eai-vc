## Environment

Set-up the environment by running:

```bash
conda create -n habitat_rearrangement_cortex python=3.8 cmake=3.14.0 -y
conda activate habitat_rearrangement_cortex
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y
pip install -r cortexbench/habitat2_vc/requirements.txt 
```

And then install local packages:
```bash
git submodule update --init --recursive
pip install -e ./third_party/habitat2/habitat-lab
pip install -e ./third_party/habitat2/habitat-baselines
pip install -e ./vc_models
```

## Data

Download data for the rearrangement task by running:

```bash
ln -s ../../data data
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path data/
curl -o data/default.physics_config.json https://raw.githubusercontent.com/facebookresearch/habitat-sim/main/data/default.physics_config.json
```
