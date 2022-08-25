# Install
create conda env, and then activate it  
`conda create --name imitation python=3.7`  
`conda activate imitation`

Install Andrew's `rl-utils` toolboox

`git clone git@github.com:ASzot/rl-utils.git`  
`cd rl-utils`  
`python setup.py develop`

Clone this repo  `git clone git@github.com:fairinternal/imitation_learning.git`  
`cd imitation_learning`   
`python setup.py develop`

Install causal world repo (they have a pip package as well, but their dependencies require an older pybullet version, 
which brings along dependencies that cause issues, so I installed from source):
`git clone https://github.com/rr-learning/CausalWorld.git`  
`cd CausalWorld`  
edit setup.py and remove the version number from pybullet (such that it just installs the latest)  
run `python setup.py develop`

# Run
* `python imitation-learning/run.py +ppo=pointmass`

# Linting 
* pip install git+https://github.com/psf/black
* `black {source_file_or_directory}`
