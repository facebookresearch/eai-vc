import argparse
import os
import os.path as osp
import uuid

try:
    import libtmux
except ImportError:
    libtmux = None
from omegaconf import OmegaConf

from rl_utils.plotting.wb_query import query_s

RUNS_DIR = "data/log/runs"


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sess-id",
        type=int,
        default=-1,
        help="tmux session id to connect to. If unspec will run in current window",
    )
    parser.add_argument(
        "--sess-name", default=None, type=str, help="tmux session name to connect to"
    )
    parser.add_argument("--proj-dat", type=str, default=None)
    parser.add_argument(
        "--group-id",
        type=str,
        default=None,
        help="If not assigned then a randomly assigned one is generated.",
    )
    parser.add_argument(
        "--run-single",
        action="store_true",
        help="""
            If true, will run all commands in a single pane sequentially. This
            will chain together multiple runs in a cmd file rather than run
            them sequentially.
    """,
    )
    parser.add_argument(
        "--cd",
        default="-1",
        type=str,
        help="""
            String of CUDA_VISIBLE_DEVICES. A value of "-1" will not set
            CUDA_VISIBLE_DEVICES at all.
            """,
    )
    parser.add_argument("--cfg", type=str, default=None)

    # MULTIPROC OPTIONS
    parser.add_argument("--mp-offset", type=int, default=0)
    parser.add_argument("--pt-proc", type=int, default=-1)

    # SLURM OPTIONS
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument(
        "--slurm-no-batch",
        action="store_true",
        help="""
            If specified, will run with srun instead of sbatch
        """,
    )
    parser.add_argument(
        "--skip-env",
        action="store_true",
        help="""
            If true, will not export any environment variables from config.yaml
            """,
    )
    parser.add_argument(
        "--speed",
        action="store_true",
        help="""
            SLURM optimized for maximum CPU usage.
            """,
    )
    parser.add_argument(
        "--st", type=str, default=None, help="Slum parition type [long, short]"
    )
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help="""
            Slurm time limit. "10:00" is 10 minutes.
            """,
    )
    parser.add_argument(
        "--c",
        type=str,
        default="7",
        help="""
            Number of cpus for SLURM job
            """,
    )
    parser.add_argument(
        "--g",
        type=str,
        default="1",
        help="""
            Number of gpus for SLURM job
            """,
    )
    parser.add_argument(
        "--ntasks",
        type=str,
        default="1",
        help="""
            Number of processes for SLURM job
            """,
    )

    return parser


def add_on_args(spec_args):
    spec_args = ['"' + x + '"' if " " in x else x for x in spec_args]
    return " ".join(spec_args)


def get_cmds(rest, args):
    cmd = rest[0]
    if len(rest) > 1:
        rest = " ".join(rest[1:])
    else:
        rest = ""

    if ".cmd" in cmd:
        with open(cmd) as f:
            cmds = f.readlines()
    else:
        cmds = [cmd]

    cmds = list(filter(lambda x: not (x.startswith("#") or x == "\n"), cmds))
    return [f"{cmd.rstrip()} {rest}" for cmd in cmds]


def get_tmux_window(sess_name, sess_id):
    if libtmux is None:
        raise ValueError("Must install libtmux to use auto tmux capability")
    server = libtmux.Server()

    if sess_name is None:
        sess = server.get_by_id("$%i" % sess_id)
    else:
        sess = server.find_where({"session_name": sess_name})
    if sess is None:
        raise ValueError("invalid session id")

    return sess.new_window(attach=False, window_name="auto_proc")


def as_list(x, max_num):
    if isinstance(x, int):
        return [x for _ in range(max_num)]
    x = x.split("|")
    if len(x) == 1:
        return [x[0] for _ in range(max_num)]
    return x


def get_cmd_run_str(cmd, args, cmd_idx, num_cmds, proj_cfg):
    conda_env = proj_cfg["conda_env"]
    python_path = osp.join(osp.expanduser("~"), "miniconda3", "envs", conda_env, "bin")
    python_path = proj_cfg.get("conda_path", python_path)

    ntasks = as_list(args.ntasks, num_cmds)
    g = as_list(args.g, num_cmds)
    c = as_list(args.c, num_cmds)

    if args.st is None:
        env_vars = " ".join(proj_cfg["add_env_vars"])
        return f"{env_vars} {cmd}"
    else:
        ident = str(uuid.uuid4())[:8]
        log_file = osp.join(RUNS_DIR, ident) + ".log"
        cmd = cmd.replace("$SLURM_ID", ident)

        if not args.slurm_no_batch:
            run_file, run_name = generate_slurm_batch_file(
                log_file,
                ident,
                python_path,
                cmd,
                args.st,
                ntasks[cmd_idx],
                g[cmd_idx],
                c[cmd_idx],
                args,
                proj_cfg,
            )
            return f"sbatch {run_file}"
        else:
            srun_settings = (
                f"--gres=gpu:{args.g} "
                + f"-p {args.st} "
                + f"-c {args.c} "
                + f"-J {ident} "
                + f"-o {log_file}"
            )

            # This assumes the command begins with "python ..."
            return f"srun {srun_settings} {python_path}/{cmd}"


def sub_wb_query(cmd, args, proj_cfg):
    parts = cmd.split("&")
    if len(parts) < 3:
        return [cmd]

    new_cmd = [parts[0]]
    parts = parts[1:]

    for i in range(len(parts)):
        if i % 2 == 0:
            wb_query = parts[i]
            result = query_s(wb_query, proj_cfg, verbose=False)
            if len(result) == 0:
                raise ValueError(f"Got no response from {wb_query}")
            sub_vals = []
            for match in result:
                if len(match) > 1:
                    raise ValueError(f"Only single value query supported, got {match}")
                sub_val = list(match.values())[0]
                sub_vals.append(sub_val)

            new_cmd = [c + sub_val for c in new_cmd for sub_val in sub_vals]
        else:
            for j in range(len(new_cmd)):
                new_cmd[j] += parts[i]

    return new_cmd


def log(s, args):
    print(s)


def split_cmd(cmd):
    cmd_parts = cmd.split(" ")
    ret_cmds = [[]]
    for cmd_part in cmd_parts:
        prefix = ""
        if "=" in cmd_part:
            prefix, cmd_part = cmd_part.split("=")
            prefix += "="

        if "," in cmd_part:
            ret_cmds = [
                ret_cmd + [prefix + split_part]
                for ret_cmd in ret_cmds
                for split_part in cmd_part.split(",")
            ]
        else:
            ret_cmds = [ret_cmd + [prefix + cmd_part] for ret_cmd in ret_cmds]
    return [" ".join(ret_cmd) for ret_cmd in ret_cmds]


def execute_command_file(run_cmd, args, proj_cfg):
    if not osp.exists(RUNS_DIR):
        os.makedirs(RUNS_DIR)

    cmds = get_cmds(run_cmd, args)

    # Sub in W&B args
    cmds = [c for cmd in cmds for c in sub_wb_query(cmd, args, proj_cfg)]

    # Split the commands.
    cmds = [c for cmd in cmds for c in split_cmd(cmd)]

    n_cmds = len(cmds)

    # Add on the project data
    if args.proj_dat is not None:
        proj_data = proj_cfg.get("proj_data", {})
        for k in args.proj_dat.split(","):
            cmds = [cmd + " " + proj_data[k] for cmd in cmds]

    add_all = proj_cfg.get("add_all", None)
    if add_all is not None:
        cmds = [cmd + " " + add_all for cmd in cmds]

    # Sub in variables
    if "base_data_dir" in proj_cfg:
        cmds = [cmd.replace("$DATA_DIR", proj_cfg["base_data_dir"]) for cmd in cmds]
    if args.group_id is None:
        group_ident = str(uuid.uuid4())[:8]
    else:
        group_ident = args.group_id
    print(f"Assigning group ID {group_ident}")
    cmds = [cmd.replace("$GROUP_ID", group_ident) for cmd in cmds]
    cmds = [cmd.replace("$CMD_RANK", str(rank_i)) for rank_i, cmd in enumerate(cmds)]

    if args.pt_proc != -1:
        pt_dist_str = f"MULTI_PROC_OFFSET={args.mp_offset} python -u -m torch.distributed.launch --use_env --nproc_per_node {args.pt_proc} "

        def make_dist_cmd(x):
            parts = x.split(" ")
            runf = None
            for i, part in enumerate(parts):
                if ".py" in part:
                    runf = i
                    break

            if runf is None:
                raise ValueError("Could not split command")

            rest = " ".join(parts[runf:])
            return pt_dist_str + rest

        cmds[0] = make_dist_cmd(cmds[0])

    DELIM = " ; "

    cd = as_list(args.cd, n_cmds)

    if args.sess_id == -1 and args.sess_name is None:
        if args.st is not None:
            for cmd_idx, cmd in enumerate(cmds):
                run_cmd = get_cmd_run_str(cmd, args, cmd_idx, n_cmds, proj_cfg)
                log(f"Running {run_cmd}", args)
                os.system(run_cmd)
        elif args.run_single:
            cmds = [get_cmd_run_str(x, args, 0, 1, proj_cfg) for x in cmds]
            exec_cmd = DELIM.join(cmds)

            log(f"Running {exec_cmd}", args)
            os.system(exec_cmd)

        elif n_cmds == 1:
            exec_cmd = get_cmd_run_str(cmds[0], args, 0, n_cmds, proj_cfg)
            if cd[0] != "-1":
                exec_cmd = "CUDA_VISIBLE_DEVICES=" + cd[0] + " " + exec_cmd
            log(f"Running {exec_cmd}", args)
            os.system(exec_cmd)
        else:
            raise ValueError("Running multiple jobs. You must specify tmux session id")
    else:
        if args.run_single:
            cmds = DELIM.join(cmds)
            cmds = [cmds]

        for cmd_idx, cmd in enumerate(cmds):
            new_window = get_tmux_window(args.sess_name, args.sess_id)

            log("running full command %s\n" % cmd, args)

            run_cmd = get_cmd_run_str(cmd, args, cmd_idx, n_cmds, proj_cfg)

            # Send the keys to run the command
            if args.st is None:
                last_pane = new_window.attached_pane
                last_pane.send_keys(run_cmd, enter=False)
                pane = new_window.split_window(attach=False)
                pane.set_height(height=50)
                pane.send_keys("source deactivate")

                if "conda_env" in proj_cfg:
                    pane.send_keys("source activate " + proj_cfg["conda_env"])
                pane.enter()
                if cd[cmd_idx] != "-1":
                    pane.send_keys("export CUDA_VISIBLE_DEVICES=" + cd[cmd_idx])
                    pane.enter()
                else:
                    pane.send_keys(run_cmd)

                pane.enter()
            else:
                pane = new_window.split_window(attach=False)
                pane.set_height(height=10)
                pane.send_keys(run_cmd)

        log("everything should be running...", args)


def generate_slurm_batch_file(
    log_file, ident, python_path, cmd, st, ntasks, g, c, args, proj_cfg
):
    ignore_nodes_s = ",".join(proj_cfg.get("slurm_ignore_nodes", []))
    if len(ignore_nodes_s) != 0:
        ignore_nodes_s = "#SBATCH -x " + ignore_nodes_s

    add_options = [ignore_nodes_s]
    if args.time is not None:
        add_options.append(f"#SBATCH --time={args.time}")
    if args.comment is not None:
        add_options.append(f'#SBATCH --comment="{args.comment}"')
    add_options = "\n".join(add_options)

    python_parts = cmd.split("python")
    has_python = False
    if len(python_parts) > 1:
        cmd = "python" + python_parts[1]
        has_python = True

    if not args.skip_env:
        env_vars = proj_cfg.get("add_env_vars", [])
        env_vars = [f"export {x}" for x in env_vars]
        env_vars = "\n".join(env_vars)

    cpu_options = "#SBATCH --cpus-per-task %i" % int(c)
    if args.speed:
        cpu_options = "#SBATCH --overcommit\n"
        cpu_options += "#SBATCH --cpu-freq=performance\n"

    if has_python:
        run_cmd = python_path + "/" + cmd
        requeue_s = "#SBATCH --requeue"
    else:
        run_cmd = cmd
        requeue_s = ""

    fcontents = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --gres gpu:%i
%s
#SBATCH --nodes 1
#SBATCH --signal=USR1@600
#SBATCH --ntasks-per-node %i
%s
#SBATCH -p %s
%s

export MULTI_PROC_OFFSET=%i
%s

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
srun %s"""
    job_name = ident
    log_file_loc = "/".join(log_file.split("/")[:-1])
    fcontents = fcontents % (
        job_name,
        log_file,
        int(g),
        cpu_options,
        int(ntasks),
        requeue_s,
        st,
        add_options,
        args.mp_offset,
        env_vars,
        run_cmd,
    )
    job_file = osp.join(log_file_loc, job_name + ".sh")
    with open(job_file, "w") as f:
        f.write(fcontents)
    return job_file, job_name


def full_execute_command_file():
    parser = get_arg_parser()
    args, rest = parser.parse_known_args()
    if args.cfg is None:
        proj_cfg = {}
    else:
        proj_cfg = OmegaConf.load(args.cfg)

    execute_command_file(rest, args, proj_cfg)
