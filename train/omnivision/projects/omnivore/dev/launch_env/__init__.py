"""
Contains defaults for launching jobs
"""

import os

from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra.core.override_parser.overrides_parser import OverridesParser
from omnivision.utils.env import get_cluster_type


CLUSTER_TYPES = [
    "prod",
    "fair",
    "aws",
]
# For some reason can't make multiple BasicSweeper objects in one job...
SWEEPER = BasicSweeper(max_batch_size=None)


PROD_CLUSTER_DICT = {
    "ash": {
        "entitlement": "gpu_fair",
        "resource_shortcut": "bbv",
    },
    "atn": {
        "entitlement": "bigbasin_atn_fair",
        "resource_shortcut": "bbv",
    },
    "pnb": {
        # multiple hostgroups in this entitlement
        "entitlement": "fair_gpu_pnb",
        "resource_shortcut": "bbv",
    },
    "prn": {
        "entitlement": "default_prn_gpu",
        "resource_shortcut": "zion2s",
    },
    "ncg": {
        "entitlement": "default_ncg",
        "resource_shortcut": "zion2s",
    },
    "dpnb": {
        # multiple hostgroups in this entitlement
        "entitlement": "default_pnb_gpu",
        "resource_shortcut": "bbv",
    },
}


def is_sweep_file(cfg_fpath: str):
    if cfg_fpath.endswith(".txt") or cfg_fpath.endswith(".sweep"):
        return True
    return False


def parse_config_name(config_path):
    assert config_path.startswith("config/")
    config_splits = config_path.split("/")
    meta_exp = config_splits[-2]
    exp = config_splits[-1]
    exp = os.path.splitext(exp)[0]
    return {"exp": exp, "meta_exp": meta_exp}


def read_txt_file_with_comments(fpath):
    lines = []
    with open(fpath, "r") as fin:
        for line in fin:
            args = line.split("#", 1)[0].strip()
            if len(args) == 0:
                continue
            lines.append(args)
    return lines


def read_txt_overrides(fpath, arg_opts):
    """Read cli from file into a string."""
    clis = read_txt_file_with_comments(fpath)
    if arg_opts:
        for arg_opt in arg_opts:
            clis.append(f"{arg_opt[0]}={arg_opt[1]}")
    # Get sweep parameters
    parser = OverridesParser.create()
    overrides = parser.parse_overrides(clis)
    run_args = SWEEPER.split_arguments(overrides, max_batch_size=None)[0]
    res = []
    for run_arg in run_args:
        res.append([el.split("=") for el in run_arg])
    return res


def get_full_exp_name(
    config_path,
    arg_opts,
    hydra_overrides=None,
    prefix="",
):
    exp_suffix = parse_config_name(config_path)["exp"]
    exp_suffix = prefix + exp_suffix
    if is_sweep_file(config_path):
        all_arg_opts = read_txt_overrides(config_path, arg_opts)
    elif arg_opts:
        # The assumption is arg_opts will never need to be split for sweeps.
        # For sweeping stuff, use the txt files
        all_arg_opts = [arg_opts]
    else:
        all_arg_opts = [[]]
    # At this point arg_opts must be a list of configs, for sweeping. It will return
    # a list of overrides and exp suffixes
    all_exp_suffix, all_hydra_overrides = [], []
    if hydra_overrides is None:
        hydra_overrides = {}  # for easy merge
    for i, arg_opts in enumerate(all_arg_opts):
        all_exp_suffix.append(exp_suffix + f"/{i}")
        all_hydra_overrides.append({**hydra_overrides, **dict(arg_opts)})
    return all_exp_suffix, all_hydra_overrides


def get_exp_dir(args):
    cluster_type = get_cluster_type()
    username = os.environ["USER"]
    if cluster_type == "fair":
        LOCAL_EXP_DIR = args.local_exp_dir
        EXP_DIR = "/checkpoint/${USER}/omnivision_omnivore/"
    elif cluster_type == "aws":
        LOCAL_EXP_DIR = args.local_exp_dir
        EXP_DIR = "/fsx-omnivore/${USER}/omnivision_omnivore/"
    elif cluster_type == "prod":
        LOCAL_EXP_DIR = args.local_exp_dir
        EXP_DIR = (
            f"manifold://{args.manifold_bucket}/tree/{username}/omnivision_omnivore/"
        )
    elif cluster_type == "oss":
        LOCAL_EXP_DIR = args.local_exp_dir
        EXP_DIR = "${HOME}/omnivision_omnivore/"
    elif cluster_type == "rsc":
        LOCAL_EXP_DIR = f"/tmp/{username}/omnivision_omnivore"
        EXP_DIR = f"/checkpoint/omniscale_ugc/{username}/omnivision_omnivore/"

    # Clear out the local exp dir
    # print(f"Deleting tmp folder: {LOCAL_EXP_DIR}")
    # shutil.rmtree(LOCAL_EXP_DIR, ignore_errors=True)

    return EXP_DIR, LOCAL_EXP_DIR
