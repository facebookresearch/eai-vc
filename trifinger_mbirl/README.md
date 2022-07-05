# Model-based IRL for the TriFinger platform
Code adapated from [the LearningtoLearn respository](https://github.com/facebookresearch/LearningToLearn/tree/main/mbirl).

## Running mbirl experiments with one demonstration

Run the script `trifinger_mbirl/run_mbirl.py` with the following command:
```
python trifinger_mbirl/run_mbirl.py --no_wandb --file_path demos/difficulty-1/demo-0000.npz
```

This will launch an mbirl run with one example demonstration (`--file_path`) without logging to wandb (--no_wandb), with the default parameters defined
in `run_mbirl.py`. These parameters can be changed by running this script with various args. By default, all output logs and plots will be saved in `trifinger_mbirl/logs/runs/`. 

## Running mbirl experiments with more than one demonstration
[In progress]
