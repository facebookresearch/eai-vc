
## Dependencies

### Install my fork of the `trifinger_simulation` package
  - This is the official pyBullet simulation of the TriFinger robot
  - To install, first clone [my fork of the package](https://github.com/ClaireLC/trifinger_simulation)
  ``` git clone https://github.com/ClaireLC/trifinger_simulation.git ```
  - Then, follow the installation [instructions in their documentation](https://open-dynamic-robot-initiative.github.io/trifinger_simulation/getting_started/installation.html)
    - On Linux, there should be no issues following the instructions as is; all the pip packages in `requirements.txt` should install with no issues.
    - On my M1 mac, I had to  first create a conda env with python 3.8 and install each of the packages in `requirements.txt` one-by-one (except the`pin` package)
      - The `pin` package only runs on Linux. For Mac, use conda to install `pinocchio`: `conda install pinocchio -c conda-forge`

## Move cube:

Move cube with impedance controller, fixed contact points, pre-computed finger trajectories. Cube is initialized at random poses near center of arena, and goal poses are randomly sampled. The current implementation only considers goal position (does not try to move cube to goal orientation). 

```
python sim_move_cube.py -v
```

## Demonstration data structure

Demonstration data is stored in `demo-*.npz` files, as a list of observations dicts for each timestep. See `scripts/viz_sim_log.py` for an example of how to load and plot demonstration trajectories. See `scripts/viz_sim_images.py` for an example of how to access observation images and create a .mp4 video with them.

Both these scripts take the path to a `demo-*.npz` file as an argument: `python scripts/viz_sim_*.py </path/to/data.npz>`

The structure of the observation dicts (per timestep) are as follows:
```
obs_dict = {
            "t": time step,
            "robot_observation": {
                                "position": joint positions,
                                "velocity": joint velocities,
                                "torque": joint torques
                                },
            "object_observation": {
                                  "position": object position,
                                  "orientation": object quaternion,
                                  },
            "camera_observation": {
                                  "camera60" : {"image": rgb image (270, 270, 3), "timestamp": camera timestamp},
                                  "camera180": {"image": rgb image (270, 270, 3), "timestamp": camera timestamp},
                                  "camera300": {"image": rgb image (270, 270, 3), "timestamp": camera timestamp},
                                  },
            "policy": {
                      "controller": {
                                    "ft_pos_cur": fingertip position - actual,
                                    "ft_pos_des": fingertip position - desired,
                                    }
                      },
            "desired_goal": goal pose dict,
            "achieved_goal": {
                             "position_error": L2 distance from current to goal object position,
                             "orientation_error": error between current to goal object orientation,
                             }
            "action": {
                      "delta_ftpos": delta fingertip positions,
                      "delta_q": delta joint positions
                      }
           }
```
