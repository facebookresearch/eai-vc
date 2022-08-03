# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

import wandb


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (imgs1, extra_imgs1, imgs2, extra_imgs2, offsets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        imgs1 = imgs1.to(device, non_blocking=True)
        extra_imgs1 = extra_imgs1.to(device, non_blocking=True)
        imgs2 = imgs2.to(device, non_blocking=True)
        extra_imgs2 = extra_imgs2.to(device, non_blocking=True)
        offsets = offsets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            (loss1, loss2), _, _ = model(
                imgs1, extra_imgs1, imgs2, extra_imgs2, offsets, args.mask_ratio1, args.mask_ratio2
            )
            loss = (1 - args.loss_weight) * loss1 + args.loss_weight * loss2

        loss1_value = loss1.item()
        loss2_value = loss2.item()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss1=loss1_value, loss2=loss2_value, loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss1_value_reduce = misc.all_reduce_mean(loss1_value)
        loss2_value_reduce = misc.all_reduce_mean(loss2_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if (data_iter_step + 1) % accum_iter == 0 and misc.get_rank() == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            wandb.log({"train_loss1": loss1_value_reduce,
                       "train_loss2": loss2_value_reduce,
                       "train_loss": loss_value_reduce,
                       "lr": lr}, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
