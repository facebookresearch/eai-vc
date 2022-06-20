
## Dependencies

### `trifinger_simulation` package
  - This is the official pyBullet simulation of the TriFinger robot
  - To install, follow the [instructions in their documentation](https://open-dynamic-robot-initiative.github.io/trifinger_simulation/getting_started/installation.html)
    - On Linux, there should be no issues following the instructions as is; all the pip packages in `requirements.txt` should install with no issues.
    - On my M1 mac, I had to  first create a conda env with python 3.8 and install each of the packages in `requirements.txt` one-by-one (except the`pin` package)
      - The `pin` package only runs on Linux. For Mac, use conda to install `pinocchio`: `conda install pinocchio -c conda-forge`

## Move cube:

Move cube with impedance controller, fixed contact points, pre-computed finger trajectories. Cube is initialized at random poses near center of arena, and goal poses are randomly sampled. The current implementation only considers goal position (does not try to move cube to goal orientation). 

```
python sim_move_cube.py -v
```
