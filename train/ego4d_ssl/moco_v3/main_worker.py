#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import builtins
import math
import os
import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import datasets.imagelistdataset, datasets.pickle_path_dataset
import torchvision.models as torchvision_models

import moco.loader
import moco.builder
import moco.optimizer
import moco_vit

from mjrl.utils.logger import DataLog
from omegaconf import DictConfig, OmegaConf
from functools import partial

torchvision_model_names = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)

model_names = [
    "vit_small",
    "vit_base",
    "vit_conv_small",
    "vit_conv_base",
] + torchvision_model_names


def main_worker(gpu, ngpus_per_node, args):
    # args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    cudnn.benchmark = True
    args.environment.gpu = gpu

    # suppress printing if not master
    if args.environment.multiprocessing_distributed and args.environment.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.environment.gpu is not None:
        print("Use GPU: {} for training".format(args.environment.gpu))

    if args.environment.distributed:
        if args.environment.dist_url == "env://" and args.environment.rank == -1:
            args.environment.rank = int(os.environ["RANK"])
        if args.environment.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.environment.rank = args.environment.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.environment.dist_backend,
            init_method=args.environment.dist_url,
            world_size=args.environment.world_size,
            rank=args.environment.rank,
        )
        torch.distributed.barrier()

    # create model
    print("=> creating model '{}'".format(args.model.arch))
    assert args.model.arch in ["vit_small", "vit_base", "resnet50", "resnet18"]
    if args.model.arch.startswith("vit"):
        model = moco.builder.MoCo_ViT(
            partial(
                moco_vit.__dict__[args.model.arch],
                stop_grad_conv1=args.model.stop_grad_conv1,
            ),
            args.model.moco_dim,
            args.model.moco_mlp_dim,
            args.model.moco_t,
            args.model.load_path,
        )
    else:
        model = moco.builder.MoCo_ResNet(
            partial(
                torchvision_models.__dict__[args.model.arch], zero_init_residual=True
            ),
            args.model.moco_dim,
            args.model.moco_mlp_dim,
            args.model.moco_t,
            args.model.load_path,
        )

    # print(model)

    # infer learning rate before changing batch size
    args.optim.lr = args.optim.lr * args.optim.batch_size / 256

    if args.environment.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.model.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.environment.gpu is not None:
            torch.cuda.set_device(args.environment.gpu)
            model.cuda(args.environment.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.optim.batch_size = int(
                args.optim.batch_size / args.environment.world_size
            )
            args.environment.workers = int(
                (args.environment.workers + ngpus_per_node - 1) / ngpus_per_node
            )
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.environment.gpu]
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.environment.gpu is not None:
        torch.cuda.set_device(args.environment.gpu)
        model = model.cuda(args.environment.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # optimizer setup
    assert args.optim.optimizer in ["lars", "adamw"]
    if args.optim.optimizer == "lars":
        optimizer = moco.optimizer.LARS(
            model.parameters(),
            args.optim.lr,
            weight_decay=args.optim.weight_decay,
            momentum=args.optim.momentum,
        )
    elif args.optim.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), args.optim.lr, weight_decay=args.optim.weight_decay
        )

    scaler = torch.cuda.amp.GradScaler()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.environment.gpu)

    # Load path
    if os.path.exists(args.environment.load_path):
        print("=> loading checkpoint '{}'".format(args.environment.load_path))
        if args.environment.gpu == "":
            checkpoint = torch.load(args.environment.load_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(args.environment.gpu)
            checkpoint = torch.load(args.environment.load_path, map_location=loc)
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}'".format(args.environment.load_path))

    # optionally resume from a checkpoint
    os.makedirs(os.path.join(args.logging.ckpt_dir, args.logging.name), exist_ok=True)
    ckpt_fname = os.path.join(
        args.logging.ckpt_dir, args.logging.name, "checkpoint_{:04d}.pth"
    )
    if args.environment.resume:
        for i in range(args.optim.epochs, -1, -1):
            if os.path.exists(ckpt_fname.format(i)):
                print("=> loading checkpoint '{}'".format(ckpt_fname.format(i)))
                if args.environment.gpu == "":
                    checkpoint = torch.load(ckpt_fname.format(i))
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = "cuda:{}".format(args.environment.gpu)
                    checkpoint = torch.load(ckpt_fname.format(i), map_location=loc)
                args.optim.start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        ckpt_fname.format(i), checkpoint["epoch"]
                    )
                )
                break

    # Create logger
    logger = None
    if args.logging.log_tb and args.environment.rank == 0:
        logger = DataLog(
            wandb_user=args.logging.wandb_user,
            wandb_project=args.logging.wandb_project,
            wandb_config=OmegaConf.to_container(
                args, resolve=True, throw_on_missing=True
            ),
        )

    cudnn.benchmark = True

    # Data loading code
    trainfname = args.data.train_filelist
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.1),
        transforms.RandomApply([moco.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    assert args.data.type in ["standard", "picklepaths"]
    if args.data.type == "standard":
        train_dataset = datasets.imagelistdataset.ImageListDataset(
            trainfname,
            base_transform1=augmentation1,
            base_transform2=augmentation2,
        )
    elif args.data.type == "picklepaths":
        train_dataset = datasets.pickle_path_dataset.PicklePathsDataset(
            root_dir=trainfname,
            frameskip=args.data.frameskip,
            transforms=[
                transforms.Compose(augmentation1),
                transforms.Compose(augmentation2),
            ]
        )
    else:
        raise NotImplementedError

    if args.environment.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.optim.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.environment.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    for epoch in range(args.optim.start_epoch, args.optim.epochs):

        if args.environment.distributed:
            train_sampler.set_epoch(epoch)

        sys.stdout.flush()
        # adjust_learning_rate(optimizer, epoch, args)
        print("Train Epoch {}".format(epoch))

        # train(train_loader, model, criterion, optimizer, epoch, args, logger=logger)
        train(train_loader, model, optimizer, scaler, logger, epoch, args)

        if not args.environment.multiprocessing_distributed or (
            args.environment.multiprocessing_distributed and args.environment.rank == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.model.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename=ckpt_fname.format(epoch),
            )

            # save the log
            if logger is not None:
                logger.save_log(save_path=args.logging.ckpt_dir)

            # remove previous checkpoint if necessary to save space
            # if os.path.exists(ckpt_fname.format(epoch - 1)):
            # os.remove(ckpt_fname.format(epoch - 1))

    if logger is not None:
        logger.run.finish()


def train(train_loader, model, optimizer, scaler, logger, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    learning_rates = AverageMeter("LR", ":.4e")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger,
    )

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.model.moco_m

    for batch_i, data in enumerate(train_loader):
        # measure data loading time
        images = [data["input1"], data["input2"]]
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + batch_i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.model.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + batch_i / iters_per_epoch, args)

        if args.environment.gpu is not None:
            images[0] = images[0].cuda(args.environment.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.environment.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss, acc1, acc5 = model(images[0], images[1], moco_m)
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1, images[0].size(0))
        top5.update(acc5, images[0].size(0))

        # compute gradient and take step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_i % args.logging.print_freq == 0:
            log_step = int(
                epoch * len(train_loader.dataset) // args.optim.batch_size
                + batch_i * torch.distributed.get_world_size()
            )
            progress.display(batch_i)
            progress.log(log_step)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch == 0:
        lr = args.optim.lr
    elif epoch < args.optim.warmup_epochs:
        lr = args.optim.lr * epoch / args.optim.warmup_epochs
    else:
        lr = (
            args.optim.lr
            * 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - args.optim.warmup_epochs)
                    / (args.optim.epochs - args.optim.warmup_epochs)
                )
            )
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1.0 - 0.5 * (1.0 + math.cos(math.pi * epoch / args.optim.epochs)) * (
        1.0 - args.model.moco_m
    )
    return m


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", tbname=""):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
        sys.stdout.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def update_global_step(self, global_step):
        if self.logger is not None:
            self.logger.global_step = global_step
        else:
            pass

    def log(self, batch=None):
        self.update_global_step(batch)
        if self.logger is not None:
            scalar_dict = self.get_scalar_dict()
            for k, v in scalar_dict.items():
                self.logger.log_kv(k, v)

    def get_scalar_dict(self):
        out = {}
        for meter in self.meters:
            val = meter.avg
            tag = meter.name
            out[tag] = val
        return out

    def save_log(self, fname):
        if self.logger is not None:
            self.logger.save_log(save_path=fname)
