# CortexBench MuJoCo-based Benchmarks

The VC MuJoCo-based Benchmarks evaluate models in various benchmarks simulated with MuJoCo. Currently, few-shot visual imitation learning is the only available evaluation methodology, and the experiment design is based on the [PVR paper](https://sites.google.com/view/pvr-control).

## Unit tests
Basic unit tests that loop over relevant environments and model_zoo are available with the package. These tests check the model integrity by loading and verifying the embedding dimension. To run the tests, execute the following command:
```bash
export DISPLAY=:100  # Requires a VNC server if headless
pytest cortexbench/mujoco_vc/tests
```

## Downloading demonstration datasets
To download the demonstration datasets, create a directory for the dataset by executing the following command:
```bash
mkdir -p cortexbench/mujoco_vc/visual_imitation/data/datasets
cd cortexbench/mujoco_vc/visual_imitation/data/datasets
```
Then, download the datasets for each benchmark with the following commands:
### Adroit benchmark:
```bash
wget https://dl.fbaipublicfiles.com/eai-vc/adroit-expert-v1.0.zip
unzip adroit-expert-v1.0.zip
rm adroit-expert-v1.0.zip
```

### Metaworld benchmark:
```bash
wget https://dl.fbaipublicfiles.com/eai-vc/mujoco_vil_datasets/metaworld-expert-v1.0.zip
unzip metaworld-expert-v1.0.zip
rm metaworld-expert-v1.0.zip
```

### DeepMind Control benchmark:
```bash
wget https://dl.fbaipublicfiles.com/eai-vc/mujoco_vil_datasets/dmc-expert-v1.0.zip
unzip dmc-expert-v1.0.zip
rm dmc-expert-v1.0.zip 
```

## Running experiments
To run experiments, navigate to the `visual_imitation` subdirectory, which contains launch scripts and config files, by executing the following commands:
```bash
cd eai-vc/cortexbench/mujoco_vc/visual_imitation/
```
To spawn an array job, execute the following command or refer to the [`launch_all_jobs.sh`](./visual_imitation/launch_all_jobs.sh) script:
```bash
python hydra_launcher.py --config-name DMC_BC_config.yaml --multirun hydra/launcher=slurm \
  env=dmc_walker_stand-v1,dmc_walker_walk-v1,dmc_reacher_easy-v1,dmc_cheetah_run-v1,dmc_finger_spin-v1 \
  embedding=$(python -m vc_models)
```

By following these steps, you can run experiments on the available benchmarks and evaluate your models accordingly.
