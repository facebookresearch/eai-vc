# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import hydra
import omegaconf
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import timm

assert timm.__version__ == "0.6.5"  # version check
import timm.optim.optim_factory as optim_factory

from eaif_models.utils.wandb import setup_wandb
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from datasets.path_dataset import PathDataset

import models_mae

from engine_pretrain import train_one_epoch


@hydra.main(config_path="configs/pretrain", config_name="config")
def main(args: omegaconf.DictConfig):
    misc.setup_wandb_output_dir(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.input_size, scale=(0.2, 1.0), interpolation=3
            ),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
        ]
    )
    extra_transform = None
    if args.color_jitter:
        extra_transform = transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ],
            p=0.8,
        )

    assert (
        type(args.data_path) == omegaconf.listconfig.ListConfig
    ), "The data_path argument should be of omegaconf.listconfig.ListConfig type"
    dataset_train = PathDataset(
        args.data_path,
        transform=transform_train,
        extra_transform=extra_transform,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_offset=args.max_offset,
        randomize_views=args.randomize_views,
    )

    print("train_dataset size: {:,}".format(len(dataset_train)))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    wandb = None
    if global_rank == 0:
        wandb = setup_wandb(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = models_mae.__dict__[args.mae_model](
        decoder_max_offset=args.max_offset,
        norm_pix_loss=args.norm_pix_loss,
        independent_decoder=args.independent_decoder,
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(
        model_without_ddp, args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args=args,
            wandb=wandb,
        )
        if args.output_dir:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                model_name=("checkpoint.pth"),
            )
            if epoch % 20 == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    model_name=("checkpoint-%s.pth" % str(epoch)),
                )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    main()
