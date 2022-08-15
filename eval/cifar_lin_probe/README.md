## CIFAR-10 Classification

We will use CIFAR-10 classification as a quick and simple computer vision task for evaluating different pre-trained models. We will evaluate in both the linear probing as well as full model finetuning (TODO) settings.

To launch a linear probing experiment on the cluster with multiple models, use the following command:
```
python run_cifar_lin_probe.py -m +hydra/launcher=submitit_slurm \
  model=moco,moco_ego4d,mae_large,mae_large_ego4d,r3m,rn50_sup_imnet
```

This should launch 6 experiments in parallel and would take about 10 minutes to complete after launching. The results will be stored in `/checkpoint/yixinlin/eaif/results/${hydra.job.name}/${oc.env:USER}` as specified in the hydra config. The final results should look something like below.

![](assets/cifar_lin_probe_results.png)
