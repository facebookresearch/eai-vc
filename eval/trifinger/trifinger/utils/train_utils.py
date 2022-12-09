import os
import sys
from hydra.core.hydra_config import HydraConfig
import wandb
import torch
import numpy as np

from utils.preprocess_trajs import (
    MODEL_NAMES,
    EAIF_MODEL_NAMES,
    CUSTOM_MODEL_NAMES,
    CUSTOM_MODEL_DECODERS,
    CUSTOM_MODEL_PREFIXES,
    get_custom_model_prefix,
)
from utils.encoder_model import EncDecModel
from utils.decoder_model import DecoderModel


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

    if params_dict["exp_dir_to_resume"] is None:
        hydra_output_dir = HydraConfig.get().runtime.output_dir
    else:
        hydra_output_dir = params_dict["exp_dir_to_resume"]

    if "experiment" in HydraConfig.get().runtime.choices:
        exp_dir_path = HydraConfig.get().sweep.dir
        exp_id = os.path.basename(os.path.normpath(exp_dir_path))
        run_id = params_dict["run_id"]
        exp_str = f"{exp_id}_r-{run_id}"
    else:
        hydra_run_dir = HydraConfig.get().run.dir
        run_date_time = "_".join(hydra_run_dir.split("/")[-2:])
        exp_id = f"single_run_{run_date_time}"
        exp_str = exp_id

    demo_path = os.path.splitext(os.path.split(params_dict["demo_path"])[1])[0]
    algo = params_dict["algo"]["name"]

    return hydra_output_dir, exp_str, exp_id


def plot_loss(loss_dict, outer_i=None):
    """Log loss to wandb"""

    log_dict = {f"{k}": v for k, v in loss_dict.items()}
    if outer_i:
        log_dict["outer_i"] = outer_i
    wandb.log(log_dict)


def find_most_recent_ckpt(ckpts_dir):

    start_epoch = 0

    files_in_ckpts_dir = os.listdir(ckpts_dir)

    if len(files_in_ckpts_dir) == 0:
        ckpt_pth = None
    else:
        for item in files_in_ckpts_dir:
            if item.endswith("ckpt.pth"):
                start_epoch = max(start_epoch, int(item.split("_")[1]))
        ckpt_pth = os.path.join(ckpts_dir, "epoch_%d_ckpt.pth" % start_epoch)

    return ckpt_pth, start_epoch


def get_obs_dict_from_traj(traj, t, obj_state_type):
    """Get observation dict for forward models"""

    obs_dict = {
        "ft_state": torch.unsqueeze(torch.FloatTensor(traj["ft_pos_cur"][t]), 0),
    }

    if "mode" in traj:
        obs_dict["mode"] = traj["mode"][t]

    if obj_state_type == "na":
        return obs_dict  # No object state

    # Add object state to obs_dict
    if obj_state_type == "pos":
        o_state = traj["o_pos_cur"][t]
    elif obj_state_type == "vertices":
        o_state = traj["vertices"][t]
    elif obj_state_type in MODEL_NAMES:
        o_state = traj[obj_state_type][t]
    else:
        raise ValueError

    obs_dict["o_state"] = torch.unsqueeze(torch.FloatTensor(o_state), 0)

    return obs_dict


def parse_traj_dict(traj_dict, cost_state, obj_state_type):
    """
    Parse out relevant part of traj_dict to compare with pred_traj from mpc

    args:
        state (str): "ftpos" | "obj" | "ftpos_obj"

    return
        traj (nparray [T, state_dim]):  where each row is a state vector of the format [ftpos, o_state]
    """

    ftpos_traj = torch.Tensor(traj_dict["ft_pos_cur"])

    latent_rep_traj = torch.Tensor(traj_dict[obj_state_type])

    if cost_state == "obj":
        # TODO assuming that "obj" state means only using latent rep
        # no support for "pos" or "vertices" obj_state_type
        traj = latent_rep_traj
    elif cost_state == "ftpos":
        traj = ftpos_traj
    elif cost_state == "ftpos_obj":
        traj = torch.cat([ftpos_traj, latent_rep_traj], dim=1)
    else:
        raise ValueError("Invalid cost_state")

    return traj


def parse_pred_traj(pred_traj, state, fnum=3, mpc_use_ftpos=True):
    """
    Parse out relevant part of pred_traj from mpc for cost function, based on state for cst

    args:
        pred_traj (nparray [T, state_dim]):  where each row is a state vector of the format [ftpos, o_state]
        state (str): cost state type: "ftpos" | "obj" | "ftpos_obj"
    """

    ftpos_dim = 3 * fnum

    if mpc_use_ftpos:
        if state == "ftpos":
            return pred_traj[:, :ftpos_dim]  # First part of pred_traj
        elif state == "obj":
            assert (
                pred_traj.shape[1] > ftpos_dim
            ), "State does not include object state. Try using mpc_type = two_phase"
            return pred_traj[:, ftpos_dim:]  # Last part of pred_traj
        elif state == "ftpos_obj":
            assert (
                pred_traj.shape[1] > ftpos_dim
            ), "State does not include object state. Try using mpc_type = two_phase"
            return pred_traj
        else:
            raise ValueError(f"{state} is invalid state")
    else:
        if state == "obj":
            return pred_traj
        else:
            raise ValueError(
                f"{state} is invalid state, pred_traj does not contain ftpos"
            )


def get_traj_list(demo_stats_list, obj_state_type):
    """Given list of demo stats demo_stats_list, load demo dicts and save in traj_list"""

    traj_list = []

    for demo_stat in demo_stats_list:
        demo_dir = demo_stat["path"]

        downsample_data_path = os.path.join(demo_dir, "downsample.pth")
        if not os.path.exists(downsample_data_path):
            raise ValueError(f"{downsample_data_path} not found")

        demo_dict = torch.load(downsample_data_path)

        if obj_state_type in MODEL_NAMES:
            # Load latent state from obj_state_type.pth file
            latent_data_path = os.path.join(demo_dir, f"{obj_state_type}.pth")
            if not os.path.exists(latent_data_path):
                raise ValueError(f"{latent_data_path} not found")

            latent_data = torch.load(latent_data_path)["data"]

            demo_dict[obj_state_type] = latent_data

        traj_list.append(demo_dict)

    return traj_list


def get_traj_range(traj_list, key, traj_stats=None):

    max_val = -np.inf
    min_val = np.inf

    if key == "position_error":
        for i, traj in enumerate(traj_list):
            if traj_stats and traj_stats[i]["diff"] not in [1, 2, 3, 11, 21, 31]:
                # For getting ranges, skip non-demo trajectories
                continue
            traj_for_key = traj[key]
            if traj_for_key[-1] > 1:
                print(traj_stats[i]["id"])
                print(traj_for_key[-1])
                print(traj["o_pos_des"][-1])
                print(traj["o_pos_cur"][-1])
            max_val = max(max_val, traj_for_key[-1])
            min_val = min(min_val, traj_for_key[-1])
    else:
        for i, traj in enumerate(traj_list):
            if traj_stats and traj_stats[i]["diff"] not in [1, 2, 3, 11, 21, 31]:
                # For getting ranges, skip non-demo trajectories
                continue
            traj_for_key = traj[key]
            max_val = max(max_val, np.max(traj_for_key))
            min_val = min(min_val, np.min(traj_for_key))

    return min_val, max_val


def get_traj_range_per_dim(traj_list, key, traj_stats=None):
    """Get per-dimension ranges"""
    # Initialize max and min arrays
    min_val = np.ones(traj_list[0][key].shape[1]) * np.inf
    max_val = np.ones(traj_list[0][key].shape[1]) * -np.inf

    for i, traj in enumerate(traj_list):
        if traj_stats and traj_stats[i]["diff"] not in [1, 2, 3, 11, 21, 31]:
            # For getting ranges, skip non-demo trajectories
            continue
        traj_for_key = traj[key]
        for t in range(traj_for_key.shape[0]):
            traj_t = traj_for_key[t, :]
            max_val = np.where(traj_t > max_val, traj_t, max_val)
            min_val = np.where(traj_t < min_val, traj_t, min_val)

    return min_val, max_val


def load_decoder(obj_state_type, device="cpu"):
    if obj_state_type in ["na", "pos", "vertices"]:
        decoder = None

    # EAIF models
    elif obj_state_type in EAIF_MODEL_NAMES:
        path_to_decoder_ckpt = EAIF_MODEL_NAMES[obj_state_type]

        if path_to_decoder_ckpt is None:
            decoder = None
        else:
            if os.path.exists(path_to_decoder_ckpt):
                decoder_model_dict = torch.load(path_to_decoder_ckpt)
                in_dim = decoder_model_dict["model_state_dict"]["model.0.weight"].shape[
                    1
                ]

                # Get pretrained_rep in decoder
                pretrained_rep = decoder_model_dict["conf"].algo.pretrained_rep
                assert (
                    obj_state_type == pretrained_rep
                ), "Mismatch between decoder pretrained_rep and obj_state_type"
                decoder = DecoderModel(
                    latent_dim=in_dim,
                )
                decoder.load_state_dict(decoder_model_dict["model_state_dict"])
                decoder.eval()
                decoder.to(device)
            else:
                decoder = None

    # Custom models
    elif obj_state_type in CUSTOM_MODEL_NAMES:
        model_prefix = get_custom_model_prefix(obj_state_type)

        # encdec
        if model_prefix == "encdec":
            path_to_decoder_ckpt = CUSTOM_MODEL_NAMES[obj_state_type]

            if os.path.exists(path_to_decoder_ckpt):
                custom_model_ckpt = CUSTOM_MODEL_NAMES[obj_state_type]
                ckpt_info = torch.load(custom_model_ckpt)
                conf = ckpt_info["conf"]
                model_state_dict = ckpt_info["model_state_dict"]
                # TODO only works for orig-r3m right now.
                encdec_model = EncDecModel(
                    pretrained_rep=conf.algo.pretrained_rep,
                    freeze_pretrained_rep=conf.algo.freeze_pretrained_rep,
                    latent_dim=conf.algo.latent_dim,
                )
                encdec_model.load_state_dict(model_state_dict)

                # Just take decoder
                decoder = encdec_model.decoder_model
                decoder.eval()
                decoder.to(device)
            else:
                decoder = None

        elif model_prefix == "bc":
            path_to_decoder_ckpt = CUSTOM_MODEL_DECODERS[obj_state_type]

            if path_to_decoder_ckpt is None:
                decoder = None
            else:
                if os.path.exists(path_to_decoder_ckpt):
                    decoder_model_dict = torch.load(path_to_decoder_ckpt)
                    in_dim = decoder_model_dict["model_state_dict"][
                        "model.0.weight"
                    ].shape[1]

                    # Get pretrained_rep in decoder
                    pretrained_rep = decoder_model_dict["conf"].algo.pretrained_rep
                    assert (
                        obj_state_type == pretrained_rep
                    ), "Mismatch between decoder pretrained_rep and obj_state_type"

                    decoder = DecoderModel(latent_dim=in_dim)
                    decoder.load_state_dict(decoder_model_dict["model_state_dict"])
                    decoder.eval()
                    decoder.to(device)
                else:
                    decoder = None
        else:
            raise NameError(f"{model_prefix} is not in {CUSTOM_MODEL_PREFIXES}")

    else:
        raise NameError

    return decoder


def get_bc_obs_vec_from_obs_dict(
    obs_dict_in,
    state_type,
    goal_type,
):
    """
    Return obs vector for bc policy

    args:
        obs_dict (dict):
        obs_type (str): [
                         "goal_none", # no goal
                         "goal_cond", # goal state appended to observation
                        ]
    """

    # If obs_dict fields aren't batched (only have 1 dim), add extra dim
    # so the shape is [1, D]
    obs_dict = {}
    for k, v in obs_dict_in.items():
        if v is not None and v.dim() == 1:
            obs_dict[k] = torch.unsqueeze(v, 0)
        else:
            obs_dict[k] = v

    if state_type == "ftpos":
        state = obs_dict["ft_state"]
    elif state_type == "obj":
        state = obs_dict["o_state"]
    elif state_type == "ftpos_obj":
        state = torch.cat([obs_dict["ft_state"], obs_dict["o_state"]], dim=1)
    else:
        raise NameError("Invalid state_type")

    if goal_type == "goal_none":
        obs_vec = state
    elif goal_type == "goal_cond":
        obs_vec = torch.cat([state, obs_dict["o_goal"]], dim=1)
    elif goal_type == "goal_o_pos":
        # Use object position goal state - relative to init position of object
        obs_vec = torch.cat([state, obs_dict["o_goal_pos_rel"]], dim=1)
    else:
        raise NameError("Invalid goal_type")

    return obs_vec


def scale_to_range(x, in_range_min, in_range_max, out_range_min, out_range_max):
    # Scale x from in_range to out_range
    # Scale to be symmetric around 0 and then shift by offset
    # From https://codereview.stackexchange.com/questions/185785/scale-numpy-array-to-certain-range

    y = (x - (in_range_max + in_range_min) / 2) / (in_range_max - in_range_min)
    return y * (out_range_max - out_range_min) + (out_range_max + out_range_min) / 2
