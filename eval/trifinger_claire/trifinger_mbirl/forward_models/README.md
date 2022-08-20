# Training forward model with `train_forward_model.py`

Run the script with `algo=forward_model`, with default params. Note: For now, you need to run the script from the trifinger_claire/ directory, because 
of the way the demo_path param is defined in config.yaml. Will save outputs to trifinger_mbirl/outputs.


```
python trifinger_mbirl/forward_model/train_forward_model.py algo=forward_model
```

Parameters for forward model training are in [this Hydra config file](https://github.com/fmeier/trifinger_claire/blob/main/trifinger_mbirl/configs/algo/forward_model.yaml).

## Training and test data

The `demo_path` param in `config.yaml` specifies the file to load training and test demos from. By default, this is set to the `demos/preloaded_demos/demos_d-1_train-1_test-1_scale-100.pth` file contained in this repo.

I'll save all other demo files in my devfair home directory: `/private/home/clairelchen/projects/demos/`.
Right now, I have a file with 100 train and 20 test demos saved in (and are also uploaded in [this google drive folder](https://drive.google.com/drive/folders/1LUYd-PDc-6MA7xS21VCGUYE8Js0E2oUp?usp=sharing)):
```
/private/home/clairelchen/projects/demos/preloaded_demos/demos_d-1_train-100_test-20_scale-100.pth
```
