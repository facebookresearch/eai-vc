# Model-based IRL for the TriFinger platform
Code adapated from [the LearningtoLearn respository](https://github.com/facebookresearch/LearningToLearn/tree/main/mbirl).

## Running mbirl experiments with one demonstration

Run the script `trifinger_mbirl/run_mbirl.py` with the following command:
```
python trifinger_mbirl/run_mbirl.py --no_wandb --file_path demos/difficulty-1/demo-0000.npz
```

This will launch an mbirl run with one example demonstration (`--file_path`) without logging to wandb (--no_wandb), with the default parameters defined
in `run_mbirl.py`. These parameters can be changed by running this script with various args. By default, all output logs and plots will be saved in `trifinger_mbirl/logs/runs/`. 

Run with best parameters so far: multi-phase cost with 50 inner-loop steps (will run 1500 outer-loop steps, which takes about 15min on my local machine)
```
python trifinger_mbirl/run_mbirl.py --no_wandb --file_path demos/difficulty-1/demo-0000.npz --cost_type MPTimeDep --n_inner_iter 50
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

## Running mbirl experiments with more than one demonstration
[In progress]
