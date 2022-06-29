# rep_eval

Given a trained representation (image -> embedding), have code in this repo to automatically evaluate it on many tasks including
- CIFAR-10 classification
- Visual imitation learning
- More ...

---

### CIFAR-10 Linear Probe Eval

To eval representations with CIFAR-10 probe:
```
$ cd rep_eval
$ python rep_eval/cifar/hydra_launcher.py -m model=resnet50,resnet50_rand,moco_conv5,mae_ViT-B,r3m,moco_ego4d_100k,moco_ego4d_5m
```
Make sure to look at the config and update any logging information.

### Visual Imitation Learning

For example, to perform visual imitation learning on the Adroit Pen task using pre-trained ResNet-50 embeddings:
```
$ cd rep_eval
$ python rep_eval/visual_il/hydra_launcher.py --config-name Adroit_BC_config.yaml -m env=pen-v0 embedding=resnet50
```
Make sure to look at the config and update any logging information.