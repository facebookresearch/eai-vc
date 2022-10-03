#!/usr/bin/env python3

"""
Example command for local run:
buck run @mode/opt dev:launch_job -- --config_path config/experiments/reference/dummy_kinetics_train_slurm_fb_cpu.yaml --local

Example command for fblearner:
buck run @mode/opt dev:launch_job -- --config_path config/experiments/reference/dummy_kinetics_train_slurm_fb_gpu_multi.yaml

Example command for fblearner with sweep:
buck run @mode/opt dev:launch_job -- --config_path config/experiments/kalyanv/testing/simple_test.sweep

Example for faircluster (local):
python dev/launch_job.py --config_path config/experiments/reference/dummy_kinetics_train_slurm_gpu.yaml --local

Example for faircluster (slurm):
python dev/launch_job.py --config_path config/experiments/reference/omnivore_in1k_eval_oss.yaml --partition devlab
"""
import argparse

import copy
import getpass
import glob
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Any, List, Tuple

from hydra import compose, initialize_config_module
from launch_env import (
    get_exp_dir,
    get_full_exp_name,
    is_sweep_file,
    parse_config_name,
    PROD_CLUSTER_DICT,
    read_txt_file_with_comments,
)
from omegaconf import DictConfig, ListConfig
from omnivision.utils.env import get_cluster_type
from pathmanager_shell_utils import PathManager_exists, PathManager_ls, PathManager_rm

CLUSTER_TYPE = get_cluster_type()
print("Detected cluster type:", CLUSTER_TYPE)
BAD_NODES_LISTS = ["/fsx-omnivore/imisra/bad_nodes.txt"]


def submitit_include_exclude_nodes_to_opts(
    exclude_nodes: List[str], include_nodes: List[str]
):
    if len(exclude_nodes) == 0:
        for fpath in BAD_NODES_LISTS:
            if os.path.exists(fpath):
                exclude_nodes += read_txt_file_with_comments(fpath)
    ret = []
    if len(exclude_nodes) > 0:
        ret += ["+submitit.exclude_nodes", "[" + ",".join(exclude_nodes) + "]"]
    if len(include_nodes) > 0:
        ret += ["+submitit.include_nodes", "[" + ",".join(include_nodes) + "]"]
    return ret


def hydra_overrides_to_string(hydra_overrides):
    run_cmd = ""
    for key in hydra_overrides:
        run_cmd += f" {key}={hydra_overrides[key]} "
    return run_cmd


def get_cfg(hydra_overrides, hydrafied_config_name):
    with initialize_config_module("omnivore.config"):
        overrides = hydra_overrides_to_string(hydra_overrides).strip().split()
        # The first override is typically the experiment one, need to replace
        # that with the config name here (adding another override doesnt work
        # since it is using +)
        assert overrides[0].startswith("+experiments")
        overrides[0] = f"+experiments={hydrafied_config_name}"
        cfg = compose("defaults", overrides=overrides)
    return cfg


def launch_one_job_prod(
    args, cfg, exp_dir, hydra_overrides, hydrafied_config_name, cfg_names, exp_name=None
):

    if exp_name is None:
        exp_name = cfg_names["exp"]

    num_nodes = cfg.launcher.num_nodes
    num_gpus = cfg.launcher.gpus_per_node

    env_vars = {
        "CONFIG_SAN": hydrafied_config_name,
        "EXP_DIR": exp_dir,
        "ENTITLEMENT": PROD_CLUSTER_DICT[args.entitlement]["entitlement"],
        "RESOURCES": PROD_CLUSTER_DICT[args.entitlement]["resource_shortcut"],
        "EXP_NAME": exp_name,
        "SECURE_GROUP": args.secure_group,
        "MANIFOLD_BUCKET": args.manifold_bucket,
    }
    torchx_exp_name = exp_name.replace("/", "_")

    cmd = ""
    for key in env_vars:
        cmd += f"export {key}={env_vars[key]} "

    cmd += "&&  "

    ex_file = "train_app"

    cmd += (
        " torchx run --workspace //bento/kernels:bento_kernel_omnivision_omnivore -s"
        " flow -cfg entitlement=$ENTITLEMENT,secure_group=$SECURE_GROUP fb.dist.ddp"
        " --img bento_kernel_omnivision_omnivore -m "
        f"omnivore.{ex_file} "
        f"-j {num_nodes}x{num_gpus} -h $RESOURCES --name {torchx_exp_name} -- "
    )

    for key in hydra_overrides:
        cmd += f"{key}={hydra_overrides[key]} "

    print(
        "########################## FBLearner Command #####################################"
    )
    print(cmd)
    print(
        "##################################################################################"
    )
    subprocess.run(cmd, shell=True, check=True)


def launch_one_job_local(
    args, cfg, exp_dir, hydra_overrides, hydrafied_config_name, cfg_names, exp_name=None
):

    if exp_name is None:
        exp_name = cfg_names["exp"]

    num_nodes = cfg.launcher.get("num_nodes", 1)
    num_gpus = cfg.launcher.get("gpus_per_node", 1)

    env_vars = {
        "CONFIG_SAN": hydrafied_config_name,
        "EXP_DIR": exp_dir,
    }

    cmd = ""
    for key in env_vars:
        cmd += f"export {key}={env_vars[key]} "

    cmd += "&&  "

    ex_file = "train_app"

    cmd += (
        " torchx run --workspace //bento/kernels:bento_kernel_omnivision_omnivore"
        f" fb.dist.ddp -m omnivore.{ex_file}"
        f" -j {num_nodes}x{num_gpus} --img bento_kernel_omnivision_omnivore -- "
    )

    for key in hydra_overrides:
        cmd += f"{key}={hydra_overrides[key]} "

    print(
        "########################## Fbcode Local Command ##################################"
    )
    print(cmd)
    print(
        "##################################################################################"
    )
    subprocess.run(cmd, shell=True, check=True)


def copy_code(src_path, dst_path, args_yes):
    # Files to always ignore during the copy process
    OMIT_DIR_LIST = {
        "__pycache__",
        "dev",
        "extra_scripts",
    }
    # Files to only ignore at the root:
    # For instance do not ignore a "tests" configuration under omnivore/config
    OMIT_ROOT_DIR_LIST = {
        "docker",
        "archive",
        "docs",
        "tests",
        "hydra_plugins",
        "tmp",
        "website",
        "tutorials",
        "third_party",
        ".git",
        ".github",
        ".circleci",
    }

    if os.path.isdir(dst_path):
        print(f"Dst code dir ({dst_path}) exists.")
        if _get_user_confirmation(
            "Delete/re-copy code dir (not output dir) and run", args_yes
        ):
            PathManager_rm(dst_path)
        else:
            print("Submitting job WITHOUT copying code. You have been warned.")
            return

    def _folder_to_ignore(_curr_dir, sub_dirs):
        return [dir for dir in sub_dirs if dir in OMIT_DIR_LIST]

    os.makedirs(dst_path)
    for file_name in os.listdir(src_path):
        if file_name in OMIT_ROOT_DIR_LIST:
            continue

        src_file_path = os.path.join(src_path, file_name)
        dst_file_path = os.path.join(dst_path, file_name)
        if os.path.isdir(src_file_path):
            shutil.copytree(src_file_path, dst_file_path, ignore=_folder_to_ignore)
        else:
            shutil.copyfile(src_file_path, dst_file_path)


def launch_one_job_slurm(
    args, cfg, exp_dir, hydra_overrides, hydrafied_config_name, cfg_names, exp_name=None
):
    if exp_name is None:
        exp_name = cfg_names["exp"]

    env_vars = {
        "CONFIG_SAN": hydrafied_config_name,
        "EXP_DIR": exp_dir,
    }

    omnivision_code_dir = Path(__file__).resolve().parents[3]
    code_sandbox_dir = os.path.join(exp_dir, "code")
    copy_code(
        omnivision_code_dir,
        code_sandbox_dir,
        args.yes,
    )

    cmd = ""
    for key in env_vars:
        cmd += f"export {key}={env_vars[key]} "
    cmd += "&&  "
    cmd += f"cd {code_sandbox_dir} && "

    # TODO: Disabling torchx for SLURM until its stable. Eventually, move to torchx for slurm.
    # cmd += f"torchx run -s slurm --scheduler_args partition={args.partition},constraint={args.constraint},time={args.time},comment={args.comment},"
    # cmd += f"mail-user={user_id}@fb.com,mail-type=END dist.ddp --script ./train_app.py -j {num_nodes}x{num_gpus} "
    # cmd += f"--cpu {args.cpu} --gpu {num_gpus} --memMB {args.mem} -- "

    # Use submitit launcher
    ex_file = "train_app_submitit.py"

    # relative path of the executable from `code_sandbox_dir`
    ex_file_path = os.path.join("projects", "omnivore", ex_file)

    cmd += f"export PYTHONPATH={code_sandbox_dir}:{code_sandbox_dir}/projects/:$PYTHONPATH && "
    cmd += f"python {ex_file_path} ++submitit.partition={args.partition} "
    cmd += f"++submitit.comment={args.comment} ++submitit.timeout_hour={args.timeout_hour} ++submitit.use_cluster=true "
    if CLUSTER_TYPE != "aws":
        cmd += f"++submitit.mem_gb={args.mem_gb} "
        if args.constraint:
            cmd += f"++submitit.constraints={args.constraint} "
    submitit_name = exp_name
    if CLUSTER_TYPE == "rsc":
        submitit_name = "omniscale/" + submitit_name
    cmd += f"++submitit.cpus_per_task={args.cpu} ++submitit.name='{submitit_name}' "

    for key in hydra_overrides:
        cmd += f"{key}={hydra_overrides[key]} "

    print(
        "########################## FAIR Cluster Command ##################################"
    )
    print(cmd)
    print(
        "##################################################################################"
    )
    subprocess.run(cmd, shell=True, check=True)


def launch_one_job_oss_local(
    args, cfg, exp_dir, hydra_overrides, hydrafied_config_name, cfg_names, exp_name=None
):

    if exp_name is None:
        exp_name = cfg_names["exp"]

    env_vars = {
        "CONFIG_SAN": hydrafied_config_name,
        "EXP_DIR": exp_dir,
    }

    cmd = ""
    for key in env_vars:
        cmd += f"export {key}={env_vars[key]} "

    cmd += "&&  "
    # cmd += f"torchx run -s local_cwd dist.ddp --script ./train_app.py -j {num_nodes}x{num_gpus} -- "

    ex_file = "train_app_submitit.py submitit.use_cluster=false "

    cmd += f"python {ex_file}"

    for key in hydra_overrides:
        cmd += f"{key}={hydra_overrides[key]} "

    print(
        "########################## FAIR CLUSTER Local Command ############################"
    )
    print(cmd)
    print(
        "##################################################################################"
    )
    subprocess.run(cmd, shell=True, check=True)


def _get_user_confirmation(qs, yes=False):
    if yes:
        print(f"Assuming 'yes' to: {qs}?")
        return True
    else:
        response = input(qs + "? Say 'yes': ")
        if response.lower() == "yes":
            return True
    return False


def check_if_already_done(args, exp_dir):
    """
    Returns True if already done, in which case it won't run again. Else returns False.
    """
    if not PathManager_exists(exp_dir):
        return False
    # exp_dir exists
    print(f"Exp Dir {exp_dir} exists.")
    if args.force:
        print("Running anyway since --force was passed.")
        return False
    # exp dir exists and not force running, check if OK to delete
    # Or if it's running locally (debugging), just delete and run
    # if args.local or _get_user_confirmation("Delete it and run", args.yes):
    if _get_user_confirmation("Delete it and run", args.yes):
        PathManager_rm(exp_dir)
        return False
    return True


def launch_one_job(
    args, cfg, exp_dir, hydra_overrides, hydrafied_config_name, cfg_names, exp_name=None
):

    if check_if_already_done(args, exp_dir):
        exit(0)

    if CLUSTER_TYPE == "prod":
        if args.local:
            launch_one_job_local(
                args,
                cfg,
                exp_dir,
                hydra_overrides,
                hydrafied_config_name,
                cfg_names,
                exp_name,
            )
        else:
            launch_one_job_prod(
                args,
                cfg,
                exp_dir,
                hydra_overrides,
                hydrafied_config_name,
                cfg_names,
                exp_name,
            )
    else:
        if args.local:
            launch_one_job_oss_local(
                args,
                cfg,
                exp_dir,
                hydra_overrides,
                hydrafied_config_name,
                cfg_names,
                exp_name,
            )
        else:
            launch_one_job_slurm(
                args,
                cfg,
                exp_dir,
                hydra_overrides,
                hydrafied_config_name,
                cfg_names,
                exp_name,
            )


def delete_fpath(fpath):
    print(f"Deleting {fpath}")
    PathManager_rm(fpath)


def delete_intermediate_ckpts(
    exp_dir, args_yes, checkpoints_dir="checkpoints", last_ckpt_name="last.ckpt"
):
    final_ckpt_dir = os.path.join(exp_dir, checkpoints_dir)
    ckpts_list = PathManager_ls(final_ckpt_dir)
    # Leave the final checkpoint in there
    ckpts_list = [
        os.path.join(final_ckpt_dir, el) for el in ckpts_list if el != last_ckpt_name
    ]
    with Pool(32) as pool:
        pool.map(delete_fpath, ckpts_list)


def fbcode_cd_and_export_cmd():
    user_id = getpass.getuser()
    primary_path = f"/home/{user_id}/fbsource/fbcode/deeplearning/projects/omnivision/projects/omnivore"
    secondary_path = (
        f"/home/{user_id}/fbcode/deeplearning/projects/omnivision/projects/omnivore"
    )
    if PathManager_exists(primary_path):
        cmd = f"cd {primary_path} &&  "
    else:
        cmd = f"cd {secondary_path} &&  "
    return cmd


def flatten(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    # from https://github.com/omry/omegaconf/pull/520/files
    # This function was never pushed to the code since it is somewhat
    # buggy, however we're just using it to get num_workers keys, so
    # should be fine.
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        raise NotImplementedError(type(cfg))

    return ret


def get_num_gpus():
    try:
        nvidia_smi_output = subprocess.run(
            "nvidia-smi --list-gpus", shell=True, check=True, capture_output=True
        ).stdout.decode("utf-8")
    except subprocess.CalledProcessError:
        return 0
    if nvidia_smi_output.find("NVIDIA-SMI has failed") == 0:
        return 0
    return int(len(nvidia_smi_output.strip().splitlines()))


def get_slurm_ids(exp_dir: str) -> List[str]:
    """
    Get the jobs running on this config. Basically reads the submitit log
    dir for unique slurm IDs.
    """
    return [
        os.path.basename(el).split("_", 1)[0]
        for el in glob.glob(os.path.join(exp_dir, "submitit_logs/*_submission.sh"))
    ]


def get_jobs_to_run_per_config(args, config_path):
    assert os.path.exists(config_path)

    cfg_names = parse_config_name(config_path)

    hydra_overrides = {
        "+experiments": "$CONFIG_SAN",
        "++launcher.experiment_log_dir": "$EXP_DIR",
    }  # TODO: Change this to Ordered Dict

    EXP_DIR, LOCAL_EXP_DIR = get_exp_dir(args)

    dst_base_dir = EXP_DIR
    if args.local:
        dst_base_dir = LOCAL_EXP_DIR

    overall_exp_dir = os.path.join(dst_base_dir, config_path).replace(
        "${USER}", getpass.getuser()
    )

    full_exp_names, hydra_overrides_per_sweep = get_full_exp_name(
        config_path,
        args.opts,
        hydra_overrides,
        args.prefix,
    )
    # If sweeping, print out which run will use what config
    all_jobs_with_id = list(enumerate(hydra_overrides_per_sweep))
    if args.sweep_run_idx is not None:
        all_jobs_with_id = [
            (i, el) for i, el in all_jobs_with_id if i in args.sweep_run_idx
        ]
        full_exp_names = [
            el for i, el in enumerate(full_exp_names) if i in args.sweep_run_idx
        ]

    jobs_info = []
    for full_exp_name, (job_id, hydra_overrides) in zip(
        full_exp_names, all_jobs_with_id
    ):
        this_cfg_names = copy.deepcopy(cfg_names)
        this_cfg_names["exp"] = full_exp_name

        if is_sweep_file(config_path):
            assert "+experiments" in hydra_overrides, (
                "When using txt configs, must have the base experiment"
                "being extended, so set a +experiments."
            )
            hydrafied_config_name = hydra_overrides["+experiments"]
        else:
            hydrafied_config_name = config_path.replace("config/experiments/", "")
            hydrafied_config_name = hydrafied_config_name.replace(".yaml", "")

        cfg = get_cfg(hydra_overrides, hydrafied_config_name)

        if args.debug:
            # Set all workers keys to 0
            worker_keys = [
                el[0] for el in flatten(cfg) if el[0].endswith("num_workers")
            ]
            for key in worker_keys:
                hydra_overrides[key] = 0

        exp_dir = os.path.join(overall_exp_dir, str(job_id))
        jobs_info.append(
            {
                "cfg": cfg,
                "exp_dir": exp_dir,
                "hydra_overrides": hydra_overrides,
                "hydrafied_config_name": hydrafied_config_name,
                "cfg_names": this_cfg_names,
            }
        )
    return jobs_info


def compare_tensorboard(exp_dirs, tb_port):
    # symlink all the tensorboard event files into one folder -- otherwise
    # it splits into versions making comparisons across jobs difficult
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Running tensorboard out of {tmpdir}")
        for exp_dir in exp_dirs:
            event_files = glob.glob(
                os.path.join(glob.escape(exp_dir), "tensorboard", "events.*")
            )
            # Old Lightning style run TB files
            event_files += glob.glob(
                os.path.join(glob.escape(exp_dir), "version_*", "events.*")
            )
            # Remove the absolute path from exp_dir
            outdir = os.path.join(tmpdir, exp_dir[1:])
            os.makedirs(outdir)
            for event_file in event_files:
                subprocess.run(f"ln -s {event_file} {outdir}/", shell=True)
        tb_port_str = ""
        if tb_port is not None:
            tb_port_str = f"--port {tb_port}"
        subprocess.run(
            f"cd {tmpdir} && tensorboard {tb_port_str} --logdir .", shell=True
        )


def main():
    parser = argparse.ArgumentParser()
    username = os.environ["USER"]

    parser.add_argument("--config_path", "-c", type=str, nargs="+", required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--local", "-l", default=False, action="store_true")
    parser.add_argument("--local_num_gpus", type=int, default=get_num_gpus())
    parser.add_argument(
        "--debug",
        "-d",
        default=False,
        action="store_true",
        help=(
            "Run locally but with 1 GPU 0 workers."
            "Note that this will override --local_num_gpus. "
            "Use --local if you just want to run locally with "
            "as many GPUs/workers as you want."
        ),
    )

    parser.add_argument(
        "--delete",
        default=False,
        action="store_true",
        help="Delete the output directory for this config",
    )
    parser.add_argument(
        "--kill_slurm_jobs",
        "-k",
        default=False,
        action="store_true",
        help="Kill all running SLURM jobs corresponding to this config",
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Skips the deleting of existing output directory and runs on that.",
    )
    # Compared to --force, that is very specific to launch without deleting
    # (it will still ask a "yes/no" question to overwrite the code_dir, which you can
    # skip by setting "-y"). -y is more general -- any yes/no question will be answered as yes.
    # Which means if you run without --force, it will go ahead and delete and run
    # (since it asks the question to delete the output dir). If you specify both --force
    # and -y, then it will not delete (since --force skips that), but when it asks
    # for overwriting the code dir, it will auto answer "yes"
    parser.add_argument(
        "-y", "--yes", help="Answer yes to every question", action="store_true"
    )
    parser.add_argument(
        "--tb",
        help="Compare the configs using tensorboard. Works on FAIR/AWS/RSC",
        action="store_true",
    )
    parser.add_argument(
        "--tb_port",
        help="Port to run TB on, by default set None to let TB pick",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--tb_otherdirs",
        help="Other dirs to use in the comparison, eg from jobs by other users",
        default=[],
        type=str,
        nargs="+",
    )

    # SLURM settings
    parser.add_argument(
        "--partition",
        type=str,
        default="learn" if CLUSTER_TYPE == "rsc" else "learnlab",
    )
    parser.add_argument("--comment", type=str, default="not_filled")
    parser.add_argument("--constraint", type=str, default=None)
    parser.add_argument("--timeout_hour", type=int, default=72)
    parser.add_argument("--cpu", type=str, default=32 if CLUSTER_TYPE == "rsc" else 10)
    parser.add_argument(
        "--mem_gb", type=str, default=1900 if CLUSTER_TYPE == "rsc" else 480
    )

    # FB Cluster settings
    parser.add_argument("--entitlement", type=str, default="dpnb")
    parser.add_argument("--secure_group", type=str, default="classy_vision_team")
    parser.add_argument("--manifold_bucket", type=str, default="omnivore")
    parser.add_argument("--pnb", help="Short cut to run on pnb", action="store_true")
    parser.add_argument("--prn", help="Short cut to run on prn", action="store_true")
    parser.add_argument(
        "--local_exp_dir",
        default=f"/tmp/{username}/omnivision_omnivore",
        type=str,
    )
    parser.add_argument(
        "--exclude_nodes",
        type=str,
        default=[],
        nargs="+",
        help=(
            "List of nodes to exclude in SLURM."
            f"Will use instead of the list from {BAD_NODES_LISTS}"
        ),
    )
    parser.add_argument(
        "--include_nodes",
        type=str,
        default=[],
        nargs="+",
        help="List of nodes to include in SLURM.",
    )

    # TODO: Add support for fast builds
    # parser.add_argument("--fast_build", "-l", default=False, action="store_true",
    # help="Performs fast build for <=1node",)

    # Sweep params
    parser.add_argument(
        "--sweep_run_idx",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Set to run a specific configs within a sweep. "
            "Can specify a list space separated."
        ),
    )

    parser.add_argument(
        "--delete_intermediate_ckpts",
        help="Delete everything except last.ckpt",
        action="store_true",
    )

    parser.add_argument("--opts", default=None, nargs="+")

    args = parser.parse_args()

    if args.pnb:
        args.entitlement = "pnb"
    if args.prn:
        args.entitlement = "prn"
    if args.debug:
        args.local = True
        args.local_num_gpus = 1
        os.environ["HYDRA_FULL_ERROR"] = "1"
    args.opts = args.opts or []
    if args.local:
        if "launcher.gpus_per_node" not in args.opts:
            args.opts.extend(["launcher.gpus_per_node", f"{args.local_num_gpus}"])
        if "launcher.num_nodes" not in args.opts:
            args.opts.extend(
                [
                    "launcher.num_nodes",
                    "1",
                ]
            )
    args.opts.extend(
        submitit_include_exclude_nodes_to_opts(args.exclude_nodes, args.include_nodes)
    )
    if args.opts:
        assert len(args.opts) % 2 == 0, "Must be even"
        # Make them pairs
        args.opts = [
            [args.opts[i], args.opts[i + 1]] for i in range(0, len(args.opts), 2)
        ]
    configs_to_run = args.config_path
    print("Running the following configs: \n", "\n".join(args.config_path))
    jobs_info = []
    for config_path in configs_to_run:
        jobs_info += get_jobs_to_run_per_config(args, config_path)

    if args.kill_slurm_jobs:
        all_slurm_ids = sum([get_slurm_ids(el["exp_dir"]) for el in jobs_info], [])
        cmd = "scancel " + " ".join(all_slurm_ids)
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        return

    if args.delete:
        for job_info in jobs_info:
            if _get_user_confirmation(f"Delete {job_info['exp_dir']}", args.yes):
                PathManager_rm(job_info["exp_dir"])
        return

    if args.delete_intermediate_ckpts:
        for job_info in jobs_info:
            delete_intermediate_ckpts(job_info["exp_dir"], args.yes)
        return

    if args.tb:
        # Just running the tensorboard for comparison
        compare_tensorboard(
            [el["exp_dir"] for el in jobs_info] + args.tb_otherdirs, args.tb_port
        )
        return
    assert not args.tb_otherdirs, "Must only be set with --tb"

    # Print out the jobs to be launched
    print(f"Launching {len(jobs_info)} jobs.")
    for i, job_info in enumerate(jobs_info):
        print(f"Run {i}: [{job_info['exp_dir']}]")
        for key, val in job_info["hydra_overrides"].items():
            print(f"\t{key}: {val}")
    # Now run the job
    for job_info in jobs_info:
        launch_one_job(
            args,
            job_info["cfg"],
            job_info["exp_dir"],
            job_info["hydra_overrides"],
            job_info["hydrafied_config_name"],
            job_info["cfg_names"],
        )


if __name__ == "__main__":
    main()
