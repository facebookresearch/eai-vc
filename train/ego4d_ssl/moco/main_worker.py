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
import datasets.imagelistdataset
import torchvision.models as models

import moco.loader
import moco.builder

from mjrl.utils.logger import DataLog
from omegaconf import DictConfig, OmegaConf


def main_worker(gpu, ngpus_per_node, args):
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

    # create model
    print("=> creating model '{}'".format(args.model.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.model.arch],
        args.model.moco_dim,
        args.model.moco_k,
        args.model.moco_m,
        args.model.moco_t,
        args.model.mlp,
    )
    if args.model.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print(model)

    if args.environment.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.environment.gpu is not None:
            torch.cuda.set_device(args.environment.gpu)
            model.cuda(args.environment.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.optim.batch_size = int(args.optim.batch_size / ngpus_per_node)
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

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.environment.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.optim.lr,
        momentum=args.optim.momentum,
        weight_decay=args.optim.weight_decay,
    )

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
    if args.logging.log_tb and args.environment.gpu == 0:
        logger = DataLog(
            wandb_user=args.logging.wandb_user,
            wandb_project=args.logging.wandb_project,
            wandb_config=OmegaConf.to_container(args, resolve=True),
        )

    cudnn.benchmark = True

    # Data loading code
    trainfname = args.data.train_filelist
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if args.data.augmentations == "all":
        augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.data.imsize, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],  # not strengthened
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif args.data.augmentations == "crop":
        augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.data.imsize, scale=(0.2, 1.0)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif args.data.augmentations == "color":
        augmentation = transforms.Compose(
            [
                transforms.Resize((args.data.imsize, args.data.imsize)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],  # not strengthened
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if args.data.type == "standard":
        train_dataset = datasets.imagelistdataset.ImageListDataset(
            trainfname, transforms=augmentation
        )
    elif args.data.type == "longtail":
        train_dataset = datasets.imagelistdataset.LongTailImageListDataset(
            trainfname, transforms=augmentation, seed=args.data.seed
        )
    elif args.data.type == "uniform":
        train_dataset = datasets.imagelistdataset.UniformImageListDataset(
            trainfname,
            transforms=augmentation,
            seed=args.data.seed,
            num_images=args.data.num_images,
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
        adjust_learning_rate(optimizer, epoch, args)
        sys.stdout.flush()
        print("Train Epoch {}".format(epoch))
        train(train_loader, model, criterion, optimizer, epoch, args, logger=logger)
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
            #     os.remove(ckpt_fname.format(epoch - 1))

    if logger is not None:
        logger.run.finish()


def train(train_loader, model, criterion, optimizer, epoch, args, logger=None):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger,
    )

    # switch to train mode
    model.train()

    end = time.time()
    for batch_i, data in enumerate(train_loader):
        # measure data loading time
        images = [data["input1"], data["input2"]]
        data_time.update(time.time() - end)

        if args.environment.gpu is not None:
            images[0] = images[0].cuda(args.environment.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.environment.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.optim.lr
    if args.optim.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.optim.epochs))
    else:  # stepwise lr schedule
        for milestone in args.optim.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
