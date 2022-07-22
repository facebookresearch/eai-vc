# Model-based IRL for the TriFinger platform
Code adapated from [the LearningtoLearn respository](https://github.com/facebookresearch/LearningToLearn/tree/main/mbirl).

## Running mbirl and bc training with train.py

`train.py` will launch either bc or mbirl training, depending on the algorithm specified, with the arg `--algo = ["bc" | "mbirl"]`. The arg `--file_path` specifies the path to a `demo/split_data.json` file, which will specify what training and what test trajectories to use.

### [NEW] Launch mbirl training with two-phase mpc

We will use the same `train.py` script, but I've added a few more args to support running with and changing parameters for the two-phase mpc.

Example command to run mbirl training with two-phase model, with full-trajectory learned cost, and using both ftpos and object state in the cost function and IRL loss.
```
python trifinger_mbirl/train.py --file_path demos/data_diff-1_train-1_test-1.json --algo mbirl --cost_type Traj --n_inner_iter 10  --mpc_type two_phase --irl_loss_state ftpos_obj --cost_state ftpos_obj --action_lr 1e-4 --no_wandb
```

args:

  * `--mpc_type` (str): mpc to use. If `"ftpos"`, use the original model in `ftpos_mpc.py`. If `"two_phase"`, use the two-phase model in `two_phase_mpc.py`.
  * `--mode` (int): Used when `--mpc_type="two_phase"`. If `--mode` is not specified, will load full demo and use full two-phase model. If `--mode=1 or `--mode=2`, will load part of demo with corresponding mode and use corresponding phase of the two-phase model.
  * `--cost_state` (str): Specify state to use in cost function
      * "ftpos": Use ftpos
      * "ftpos_obj": Use ftpos and object state
      * "obj": Use only object state
  * `--irl_loss_state` (str): Specify state to use in cost function
      * "ftpos": Use ftpos
      * "ftpos_obj": Use ftpos and object state
      * "obj": Use only object state
  * `--no_wandb`: If specified, do not log to wandb
  * `--cost_type`: Learnable cost to use. Currently multi-phase costs are not supported for running with two-phase mpc.
  * `--action_lr`: Policy learning rate
  * `--cost_lr`: Cost function learning rate

### Launch mbirl training with a single train and test trajectory with this command:
```
python trifinger_mbirl/train.py --file_path demos/data_diff-1_train-1_test-1.json -a mbirl --no_wandb --cost_type MPTimeDep --n_inner_iter 50
```

This will launch an mbirl run with one example demonstration (`--file_path`) without logging to wandb (--no_wandb), with the default parameters defined
in `run_mbirl.py`. These parameters can be changed by running this script with various args. By default, all output logs and plots will be saved in `trifinger_mbirl/logs/runs/`. Will run with best parameters I've found so far: multi-phase cost with 50 inner-loop steps.

### Launch bc training
```
python trifinger_mbirl/train.py --file_path demos/data_diff-1_train-1_test-1.json -a bc --no_wandb --bc_obs_type goal_rel
```

## Plot costs

So far, this script only supports plotting the weights for the multi-phase cost. 

```
python trifinger_mbirl/plot_learned_cost.py <PATH/TO/EXP/log.pth> <-s SAVE FIG TO EXP DIR>
```

Example usage:
```
python trifinger_mbirl/plot_learned_cost.py /Users/clairelchen/projects/trifinger_claire/trifinger_mbirl/logs/runs/exp_NOID_al-0p01_cl-0p01_ct-MPTimeDep_ils-100_nii-50_noi-1500_rk-5_rw-2_th-17/log.pth -s
```
