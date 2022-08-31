import os
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import wandb


def get_exp_dir(params_dict):
    """
    Get experiment directory to save logs in, and experiment name

    args:
        params_dict: hydra config dict
    return:
        exp_dir: Path of experiment directory
        exp_str: Name of experiment run - to name wandb run
        exp_id: Experiment id - for conf.exp_id to label wandb run
    """

    hydra_output_dir = HydraConfig.get().runtime.output_dir

    if "experiment" in HydraConfig.get().runtime.choices:
        exp_dir_path = HydraConfig.get().sweep.dir
        exp_id = os.path.basename(os.path.normpath(exp_dir_path))
    else:
        hydra_run_dir = HydraConfig.get().run.dir
        run_date_time = "_".join(hydra_run_dir.split("/")[-2:])
        exp_id = f"single_run_{run_date_time}"

    run_id = params_dict["run_id"]
    demo_path = os.path.splitext(os.path.split(params_dict["demo_path"])[1])[0]
    algo = params_dict["algo"]["name"]

    exp_str = f"{exp_id}_r-{run_id}"

    return hydra_output_dir, exp_str, exp_id


def plot_loss(loss_dict, outer_i):
    """Log loss to wandb"""

    log_dict = {f"{k}": v for k, v in loss_dict.items()}
    log_dict["outer_i"] = outer_i
    wandb.log(log_dict)
