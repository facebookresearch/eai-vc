## Installation
1. Download this repositiory:
   ```
   git clone git@github.com:arjunmajum/mae-for-eai.git
   ```

1. Create a conda environment:
   ```
   conda create -n eai python=3.8 cmake=3.22.1
   ```
   ```
   conda activate eai
   ```

1. Install [pytorch](https://pytorch.org/) version `1.10.2`:
   ```
   conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 -c pytorch -c conda-forge
   ```

1. Install additional requirements:
   ```
   cd mae-for-eai
   ```
   ```
   pip install -r requirements.txt
   ```


1. Install [habitat-sim](https://github.com/facebookresearch/habitat-sim/tree/v0.2.1) version `0.2.1`:
   ```
   conda install habitat-sim==0.2.1 headless -c conda-forge -c aihabitat
   ```

1. Install [habitat-lab](https://github.com/facebookresearch/habitat-lab/tree/v0.2.1) version `0.2.1`:
   ```
   cd ..  # exit the mae-for-eai directory
   ```
   ```
   git clone --branch v0.2.1 https://github.com/facebookresearch/habitat-lab.git habitat-lab-v0.2.1
   ```
   ```
   cd habitat-lab-v0.2.1
   ```
   ```
   pip install -r requirements.txt
   ```
   ```
   python setup.py develop --all # install habitat and habitat_baselines
   ```

1. Install mae-for-eai:

   ```
   cd mae-for-eai
   ```
   ```
   python setup.py develop
   ```

1. [Optionally] install development requirements
   ```
   pip install -r optional-requirements.txt
   ```
   ```
   conda install ipython jupyterlab
   ```

1. Follow the instructions [here](https://github.com/facebookresearch/habitat-lab#data) to set up the `data/scene_datasets/` directory. Your directory structure should now look like this:
   ```
   .
   +-- habitat-lab-v0.2.1/
   |   ...
   +-- mae-for-eai/
   |   +-- data/
   |   |   +-- datasets/
   |   |   |   +-- imagenav/
   |   |   +-- scene_datasets/
   |   |   |   +-- gibson/
   |   +-- eai/
   |   ...
   ```
