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

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
except ImportError:
    xm = xmp = pl = xu = None


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if not misc.XLA_CFG["is_xla"]:
            samples = samples.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=not misc.XLA_CFG["is_xla"]):
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        else:
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
            (loss / accum_iter).backward()
            if (data_iter_step + 1) % accum_iter == 0:
                xm.reduce_gradients(optimizer)
                optimizer.step()
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if not misc.XLA_CFG["is_xla"]:
            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
        else:
            # TODO(ronghanghu) figure out a better way for logging in XLA
            # In XLA, it's too expensive to log every iteration due to the overhead
            # of moving the loss scalar from TPU to host. 
            if (data_iter_step - 1) % misc.XLA_CFG["logging_interval"] == 0:
                xm.add_step_closure(_xla_logging, args=(metric_logger, loss))

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if not misc.XLA_CFG["is_xla"]:
            # TODO(ronghanghu): add tensorboard logging in XLA mode
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _xla_logging(metric_logger, loss):
    metric_logger.update(loss=loss.item())
