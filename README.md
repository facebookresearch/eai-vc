# rep_eval

Given a trained representation (image -> embedding), have code in this repo to automatically evaluate it on many tasks including
- CIFAR-10 classification
- Imitation learning on MuJoCo environments
- More ...

---

### CIFAR-10 Linear Probe Eval

To eval representations with CIFAR-10 probe:
```
$ cd cifar
$ python hydra_launcher.py -m model=resnet50,resnet50_rand,moco_conv5,mae_ViT-B,r3m,moco_ego4d_100k,moco_ego4d_5m
```
Make sure to look at the config and update any logging information.