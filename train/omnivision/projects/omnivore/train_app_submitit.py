# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import getpass
import os
import random

import hydra
import submitit
import torch
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf
from omnivore.train_utils import makedir, register_omegaconf_resolvers

# Make work w recent PyTorch versions (https://github.com/pytorch/pytorch/issues/37377)
os.environ["MKL_THREADING_LAYER"] = "GNU"
MAIL_USERS = {
    "rgirdhar": "ALL",
}

register_omegaconf_resolvers()


class SubmititRunner(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs."""

    def __init__(self, port, cfg):
        self.cfg = cfg
        self.port = port

    def __call__(self):
        register_omegaconf_resolvers()
        job_env = submitit.JobEnvironment()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        trainer = instantiate(self.cfg.trainer, _recursive_=False)
        trainer.run()


def single_proc_run(local_rank, main_port, cfg, world_size):
    """Executes fun() on a single GPU in a multi-GPU setup."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1
    num_proc = cfg.launcher.gpus_per_node
    if num_proc == 1:
        # directly call single_proc so we can easily set breakpoints
        # mp.spawn does not let us set breakpoints
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        # Note: using "fork" below, "spawn" causes time and error regressions. Using
        # spawn changes the default multiprocessing context to spawn, which doesn't
        # interact well with the dataloaders (likely due to the use of OpenCV).
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


@hydra.main(config_path="config", config_name=None, version_base="1.2")
def main(cfg) -> None:
    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    makedir(cfg.launcher.experiment_log_dir)
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    if submitit_conf.get("log_save_dir") is None:
        submitit_dir = cfg.launcher.experiment_log_dir
        submitit_dir = os.path.join(submitit_dir, "submitit_logs")
    else:
        submitit_dir = submitit_conf.log_save_dir

    if submitit_conf.use_cluster:
        executor = submitit.AutoExecutor(folder=submitit_dir)

        job_kwargs = {
            "timeout_min": 60 * submitit_conf.timeout_hour,
            "name": submitit_conf.name,
            "slurm_partition": submitit_conf.partition,
            "gpus_per_node": cfg.launcher.gpus_per_node,
            "tasks_per_node": cfg.launcher.gpus_per_node,  # one task per GPU
            "cpus_per_task": submitit_conf.cpus_per_task,
            "nodes": cfg.launcher.num_nodes,
            "slurm_additional_parameters": {
                "exclude": " ".join(submitit_conf.get("exclude_nodes", [])),
            },
        }
        if "include_nodes" in submitit_conf:
            assert (
                len(submitit_conf["include_nodes"]) >= cfg.launcher.num_nodes
            ), "Not enough nodes"
            job_kwargs["slurm_additional_parameters"]["nodelist"] = " ".join(
                submitit_conf["include_nodes"]
            )
        username = getpass.getuser()
        if username in MAIL_USERS:
            job_kwargs["slurm_additional_parameters"][
                "mail-user"
            ] = f"{username}@fb.com"
            job_kwargs["slurm_additional_parameters"]["mail-type"] = MAIL_USERS[
                username
            ]

        if submitit_conf.get("mem_gb", None) is not None:
            job_kwargs["mem_gb"] = submitit_conf.mem_gb
        elif submitit_conf.get("mem", None) is not None:
            job_kwargs["slurm_mem"] = submitit_conf.mem

        if submitit_conf.get("constraints", None) is not None:
            job_kwargs["slurm_constraint"] = submitit_conf.constraints

        if submitit_conf.get("comment", None) is not None:
            job_kwargs["slurm_comment"] = submitit_conf.comment

        print("###################### SLURM Config ####################")
        print(job_kwargs)
        print("##########################################")
        executor.update_parameters(**job_kwargs)

        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        job = executor.submit(SubmititRunner(main_port, cfg))
        print("Submitit Job ID:", job.job_id)
    else:
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        single_node_runner(cfg, main_port)


if __name__ == "__main__":
    main()
