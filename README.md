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
