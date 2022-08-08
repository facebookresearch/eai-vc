# EAI Foundations

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/facebookresearch/eai-foundations/tree/main.svg?style=svg&circle-token=52b92efd205cf081e310aaad27be9c74a86190b3)](https://dl.circleci.com/status-badge/redirect/gh/facebookresearch/eai-foundations/tree/main)

This is the FAIR monorepo for investigating large-scale pretraining for embodied AI.

## Environment installation

```
conda env create -f eval/cifar_lin_probe/environment.yml
```

Install mujoco py:

```
conda activate cifar_linprobe

cp -r /private/home/aravraj/.mujoco ~/
cp -r /private/home/aravraj/work/Projects/mujoco_gpu/mujoco-py ~/
pip install -e ~/mujoco-py
```


```
# Install mjrl



# Install mj_envs


img = env.sim.render(width=84, height=84, depth=False, camera_name=None, device_id=-1)

```

