#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.utils.tensorboard import SummaryWriter
import hydra.utils as hydra_utils
import hydra
import submitit
import logging
import copy
from pathlib import Path
import warnings
import random
import os
import numpy as np

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

MAIN_PID = os.getpid()
SIGNAL_RECEIVED = False

log = logging.getLogger(__name__)


def update_pythonpath_relative_hydra():
    """Update PYTHONPATH to only have absolute paths."""
    # NOTE: We do not change sys.path: we want to update paths for future instantiations
    # of python using the current environment (namely, when submitit loads the job
    # pickle).
    try:
        original_cwd = Path(hydra_utils.get_original_cwd()).resolve()
    except (AttributeError, ValueError):
        # Assume hydra is not initialized, we don't need to do anything.
        # In hydra 0.11, this returns AttributeError; later it will return ValueError
        # https://github.com/facebookresearch/hydra/issues/496
        # I don't know how else to reliably check whether Hydra is initialized.
        return
    paths = []
    for orig_path in os.environ["PYTHONPATH"].split(":"):
        path = Path(orig_path)
        if not path.is_absolute():
            path = original_cwd / path
        paths.append(path.resolve())
    os.environ["PYTHONPATH"] = ":".join([str(x) for x in paths])
    log.info("PYTHONPATH: {}".format(os.environ["PYTHONPATH"]))


class Worker:
    def __call__(self, origargs):
        """TODO: Docstring for __call__.

        :args: TODO
        :returns: TODO

        """
        from main_worker import main_worker
        import numpy as np
        import torch
        import torch.nn.parallel
        import torch.optim
        import torch.multiprocessing as mp
        import torch.utils.data
        import torch.utils.data.distributed
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = True
        args = copy.deepcopy(origargs)
        np.set_printoptions(precision=3)
        if args.environment.seed == 0:
            args.environment.seed = None
        socket_name = (
            os.popen("ip r | grep default | awk '{print $5}'").read().strip("\n")
        )
        print("Setting GLOO and NCCL sockets IFNAME to: {}".format(socket_name))
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name
        # not sure if the next line is really affect anything
        # os.environ["NCCL_SOCKET_IFNAME"] = socket_name

        if args.environment.slurm:
            job_env = submitit.JobEnvironment()
            args.environment.rank = job_env.global_rank
            hostname_first_node = (
                os.popen("scontrol show hostnames $SLURM_JOB_NODELIST")
                .read()
                .split("\n")[0]
            )
            args.environment.dist_url = (
                f"tcp://{job_env.hostnames[0]}:{args.environment.port}"
            )
        else:
            args.environment.dist_url = (
                f"tcp://{args.environment.node}:{args.environment.port}"
            )
        print("Using url {}".format(args.environment.dist_url))

        print("Using url {}".format(args.environment.dist_url))

        if args.environment.seed is not None:
            random.seed(args.environment.seed)
            torch.manual_seed(args.environment.seed)
            warnings.warn(
                "You have chosen to seed training. "
                "This will turn on the CUDNN deterministic setting, "
                "which can slow down your training considerably! "
                "You may see unexpected behavior when restarting "
                "from checkpoints."
            )
        if args.environment.gpu != "":
            warnings.warn(
                "You have chosen a specific GPU. This will completely "
                "disable data parallelism."
            )

        if args.environment.dist_url == "env://" and args.environment.world_size == -1:
            args.environment.world_size = int(os.environ["WORLD_SIZE"])

        args.environment.distributed = (
            args.environment.world_size > 1
            or args.environment.multiprocessing_distributed
        )
        ngpus_per_node = torch.cuda.device_count()
        if args.environment.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.environment.world_size = ngpus_per_node * args.environment.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            main_worker(args.environment.gpu, ngpus_per_node, args)

    def checkpoint(self, *args, **kwargs) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            self, *args, **kwargs
        )  # submits to requeuing


def load_jobs(N=1000, end_after="$(date +%Y-%m-%d-%H:%M)"):
    jobs = (
        os.popen(
            f'sacct -u $USER --format="JobID,JobName,Partition,State,End,Comment" '
            f'-X -P -S "{end_after}" | tail -n {N}'
        )
        .read()
        .split("\n")
    )
    jobs_parsed = []
    for line in jobs:
        row = line.strip().split("|")
        if len(row) != 6:
            continue
        if row[0] == "JobID":
            continue
        job_id_raw, name, partition, status, end, comment = row
        job_id_comp = job_id_raw.strip().split("_")
        job_id = int(job_id_comp[0])
        try:
            if len(job_id_comp) == 2:
                sort_key = (end, job_id, int(job_id_comp[1]))
            else:
                sort_key = (end, job_id, 0)
        except ValueError:
            print("Error parsing job: ", job_id)
            continue
        jobs_parsed.append(
            [job_id_raw, name, partition, status, end, comment, sort_key]
        )
    jobs_parsed = sorted(jobs_parsed, key=lambda el: el[-1])
    return jobs_parsed


@hydra.main(config_path="./configs/moco", config_name="config", version_base="1.1")
def main(args):
    update_pythonpath_relative_hydra()
    args.logging.ckpt_dir = hydra_utils.to_absolute_path(args.logging.ckpt_dir)
    args.logging.tb_dir = hydra_utils.to_absolute_path(args.logging.tb_dir)
    args.data.train_filelist = hydra_utils.to_absolute_path(args.data.train_filelist)
    args.data.val_filelist = hydra_utils.to_absolute_path(args.data.val_filelist)

    # If job is running, ignore
    jobdets = load_jobs()
    jobnames = [j[1] for j in jobdets]
    if (args.logging.name).replace(".", "_") in jobnames and args.environment.slurm:
        print("Skipping {} because already in queue".format(args.logging.name))
        return

    # If model is trained, ignore
    ckpt_fname = os.path.join(
        args.logging.ckpt_dir, args.logging.name, "checkpoint_{:04d}.pth"
    )
    if os.path.exists(ckpt_fname.format(args.optim.epochs - 1)):
        print("Skipping {}".format(args.logging.name))
        return

    executor = submitit.AutoExecutor(
        folder=os.path.join(args.logging.submitit_dir, "{}".format(args.logging.name)),
        max_num_timeout=100,
        cluster=None if args.environment.slurm else "debug",
    )
    executor.update_parameters(
        timeout_min=args.environment.slurm_timeout,
        slurm_partition=args.environment.slurm_partition,
        cpus_per_task=args.environment.workers,
        gpus_per_node=args.environment.ngpu,
        nodes=args.environment.world_size,
        tasks_per_node=1,
        mem_gb=256,
    )
    executor.update_parameters(name=args.logging.name)
    job = executor.submit(Worker(), args)
    if not args.environment.slurm:
        job.result()


if __name__ == "__main__":
    main()
