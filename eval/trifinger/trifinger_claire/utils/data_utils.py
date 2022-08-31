import numpy as np
import matplotlib.pyplot as plt
import json
import os
import torch
import sys
import copy
import imageio

from r3m import load_r3m
import torchvision.transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, ".."))

from trifinger_mbirl.forward_models.models.forward_model import (
    get_obs_vec_from_obs_dict,
)

NON_TRAJ_KEYS = ["ft_pos_targets_per_mode"]


def get_traj_dict_from_obs_list(data, scale=1, include_image_obs=True):
    """
    Process list of observation dicts into dict of lists (trajectories for each quantity)

    args:
        data: list of observation dicts
        scale: amount to scale all distances by. by default, all distances are in meters. convert to cm with scale=100
    """
    data_keys = data[0].keys()

    if "object_observation" in data_keys:  # prev. key in obs
        position_error = np.array(
            [data[i]["achieved_goal"]["position_error"] for i in range(len(data))]
        )
        o_cur = np.array(
            [data[i]["object_observation"]["position"] for i in range(len(data))]
        )
        o_cur_ori = np.array(
            [data[i]["object_observation"]["orientation"] for i in range(len(data))]
        )
        robot_pos = np.array(
            [data[i]["robot_observation"]["position"] for i in range(len(data))]
        )
        o_des = np.array(
            [data[i]["desired_goal"]["position"] for i in range(len(data))]
        )
        o_des_ori = np.array(
            [data[i]["desired_goal"]["orientation"] for i in range(len(data))]
        )

    else:
        position_error = np.array(
            [data[i]["achieved_goal_position_error"] for i in range(len(data))]
        )
        o_cur = np.array([data[i]["object_position"] for i in range(len(data))])
        o_cur_ori = np.array([data[i]["object_orientation"] for i in range(len(data))])
        robot_pos = np.array([data[i]["robot_position"] for i in range(len(data))])
        o_des = np.array([data[i]["desired_goal"][:3] for i in range(len(data))])
        o_des_ori = np.array([data[i]["desired_goal"][3:] for i in range(len(data))])

    ft_pos_cur = np.array(
        [data[i]["policy"]["controller"]["ft_pos_cur"] for i in range(len(data))]
    )
    ft_pos_des = np.array(
        [data[i]["policy"]["controller"]["ft_pos_des"] for i in range(len(data))]
    )
    delta_ftpos = np.array([data[i]["action"]["delta_ftpos"] for i in range(len(data))])
    ft_vel_cur = np.array(
        [data[i]["policy"]["controller"]["ft_vel_cur"] for i in range(len(data))]
    )
    ft_vel_des = np.array(
        [data[i]["policy"]["controller"]["ft_vel_des"] for i in range(len(data))]
    )

    t = np.expand_dims(np.array([data[i]["t"] for i in range(len(data))]), 1)

    traj_dict = {
        "t": t,
        "o_pos_cur": scale * o_cur,
        "o_pos_des": scale * o_des,
        "o_ori_cur": scale * o_cur_ori,
        "o_ori_des": scale * o_des_ori,
        "ft_pos_cur": scale * ft_pos_cur,
        "ft_pos_des": scale * ft_pos_des,
        "ft_vel_cur": scale * ft_vel_cur,
        "ft_vel_des": scale * ft_vel_des,
        "position_error": scale * position_error,
        "delta_ftpos": scale * delta_ftpos,
        "robot_pos": robot_pos,
    }

    if include_image_obs:
        image60 = np.array(
            [
                data[i]["camera_observation"]["camera60"]["image"]
                for i in range(len(data))
            ]
        )
        image180 = np.array(
            [
                data[i]["camera_observation"]["camera180"]["image"]
                for i in range(len(data))
            ]
        )
        image300 = np.array(
            [
                data[i]["camera_observation"]["camera300"]["image"]
                for i in range(len(data))
            ]
        )
        traj_dict["image_60"] = image60
        traj_dict["image_180"] = image180
        traj_dict["image_300"] = image300

    # Mode information
    if "ft_pos_targets_per_mode" in data[-1]["policy"]:
        traj_dict["ft_pos_targets_per_mode"] = (
            scale * data[-1]["policy"]["ft_pos_targets_per_mode"]
        )

        # Add "mode"
        if "mode" not in data[0]["policy"]:
            traj_dict["mode"] = np.array(
                [
                    len(data[i]["policy"]["ft_pos_targets_per_mode"])
                    for i in range(len(data))
                ]
            )
        else:
            traj_dict["mode"] = np.array(
                [data[i]["policy"]["mode"] for i in range(len(data))]
            )

    # Object vertices
    if "object_observation" in data[0] and "vertices" in data[0]["object_observation"]:
        vertices = []
        # Flatten vertices dict at each timestep and add to vertices list
        for i in range(len(data)):
            v_wf_dict = data[i]["object_observation"]["vertices"]
            v_wf_flat = np.zeros(len(v_wf_dict) * 3)
            for k, v_wf in v_wf_dict.items():
                v_wf_flat[k * 3 : k * 3 + 3] = v_wf
            vertices.append(v_wf_flat)

        traj_dict["vertices"] = scale * np.array(vertices)

    return traj_dict


def downsample_traj_dict(traj_dict, cur_time_step=0.004, new_time_step=0.1):
    """
    Downsample each of the trajectories in traj_dict.
    Add R3M embeddings for image_60
    """

    # Load R3M
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    r3m_model = load_r3m("resnet50")  # resnet18, resnet34
    r3m_model.eval()
    r3m_model.to(device)

    every_x_steps = max(1, int(new_time_step / cur_time_step))
    num_waypoints = int(traj_dict["t"].shape[0] / every_x_steps)
    indices_to_take = np.linspace(
        0, traj_dict["t"].shape[0] - 1, num_waypoints, dtype=int
    )

    new_traj_dict = {}

    for k, traj in traj_dict.items():
        if "delta" in k:
            continue  # Need to recompute deltas for downsampled traj

        if k in NON_TRAJ_KEYS:
            new_traj_dict[k] = traj
            continue

        # new_traj = traj[::every_x_steps, :]
        new_traj = traj[indices_to_take]
        new_traj_dict[k] = new_traj

    # Compute deltas for downsampled traj
    new_delta_ftpos = np.zeros(new_traj_dict["ft_pos_cur"].shape)
    ft_pos = new_traj_dict["ft_pos_cur"]
    for t in range(ft_pos.shape[0] - 1):
        delta = ft_pos[t + 1] - ft_pos[t]
        new_delta_ftpos[t, :] = delta
    new_traj_dict["delta_ftpos"] = new_delta_ftpos

    # Compute r3m on image_60
    image_60 = new_traj_dict["image_60"]
    r3m_dim = get_r3m_img(r3m_model, image_60[0, :]).shape[0]
    image_60_r3m = np.zeros((num_waypoints, r3m_dim))
    for i in range(num_waypoints):
        r3m_i = get_r3m_img(r3m_model, image_60[i, :]).cpu().numpy()
        image_60_r3m[i, :] = r3m_i
    new_traj_dict["image_60_r3m"] = image_60_r3m

    return new_traj_dict


def get_traj_mode(traj_dict, mode):
    """Parse out part of trajectory with corresponding mode"""

    assert mode in [1, 2], "Invalid mode"

    new_traj_dict = {}

    indices_to_take = np.where(traj_dict["mode"] == mode)[0]
    for k, traj in traj_dict.items():
        if k in NON_TRAJ_KEYS:
            new_traj_dict[k] = traj
            continue

        new_traj = traj[indices_to_take]
        new_traj_dict[k] = new_traj

    return new_traj_dict


def crop_traj_dict(traj_dict, crop_range):
    """crop_range: [crop_min, crop_max]"""

    if crop_range[0] is None:
        crop_min = 0
    else:
        crop_min = crop_range[0]

    new_traj_dict = {}

    for k, traj in traj_dict.items():
        if crop_range[1] is None:
            crop_max = traj.shape[0]
        else:
            crop_max = crop_range[1]

        if k in NON_TRAJ_KEYS:
            new_traj_dict[k] = traj
            continue

        new_traj = traj[crop_min:crop_max, :]
        new_traj_dict[k] = new_traj

    return new_traj_dict


def plot_traj(title, save_path, d_list, data_dicts, plot_timestamp=None):
    """
    Plot trajectories

    data_dicts = {
                 "label_1": {"y": y data, "x": x data (optional), "marker": marker string (optional)],},
                 "label_2": {"y": y data, "x"},
                 ...
                 }
    """

    plt.figure(figsize=(10, 10), dpi=200)
    plt.subplots_adjust(hspace=1)
    plt.suptitle(title)

    k = 0
    for i, d in enumerate(d_list):
        k += 1
        plt.subplot(len(d_list), 1, k)
        if len(d_list) > 1:
            plt.title(d)

        for label, data in data_dicts.items():

            num_steps = data["y"].shape[0]

            if "x" in data:
                x = data["x"]
            else:
                x = list(range(num_steps))

            if "marker" in data:
                marker = data["marker"]
            else:
                marker = None

            plt.plot(x, data["y"][:, i], marker=marker, label=label)

        if plot_timestamp is not None:
            plt.axvline(x=plot_timestamp, ls="--", c="k", lw=1)

    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def load_trajs(traj_load_info, save_path=None, scale=1, mode=None):
    """
    Load train and test trajectories from traj_load_info

    Args
        traj_load_info: a dict in the following format:
                        {
                            "demo_dir"   : top-level directory containing demos ("demos/"),
                            "difficulty" : difficulty level (1),
                            "train_demos": list of demo ids for training ([0,1]),
                            "test_demos" : list of demo ids for testing ([5]),
                        }
        save_path (str): If specified, save demo info in save_path
        scale: amount to scale distances by
        mode (int): 1 or 2; if specified, only return part of trajectory with this mode
    """

    def get_demo_path(demo_dir, diff, demo_id):
        demo_path = os.path.join(
            demo_dir, f"difficulty-{diff}", f"demo-{demo_id:04d}.npz"
        )
        return demo_path

    downsample_time_step = 0.2
    demo_dir = traj_load_info["demo_dir"]

    train_trajs = []
    test_trajs = []

    train_demo_stats = []
    test_demo_stats = []

    # Load and downsample test trajectories for each difficulty
    for i, diff in enumerate(traj_load_info["difficulty"]):
        train_demo_ids = traj_load_info["train_demos"][i]
        test_demo_ids = traj_load_info["test_demos"][i]

        for demo_id_list, traj_list, stats_list in [
            [train_demo_ids, train_trajs, train_demo_stats],
            [test_demo_ids, test_trajs, test_demo_stats],
        ]:

            for demo_id in demo_id_list:
                demo_path = get_demo_path(demo_dir, diff, demo_id)

                demo_stats = {"path": demo_path, "diff": diff, "id": demo_id}
                stats_list.append(demo_stats)

                data = np.load(demo_path, allow_pickle=True)["data"]
                traj_original = get_traj_dict_from_obs_list(data, scale=scale)

                # Full trajectory, downsampled
                traj = downsample_traj_dict(
                    traj_original, new_time_step=downsample_time_step
                )

                if mode is not None:
                    traj = get_traj_mode(traj, mode)

                traj_list.append(traj)

    print(f"Loaded {len(train_trajs)} training demos")
    print(f"Loaded {len(test_trajs)} test demos")

    # Save demo info (train and test demos)
    if save_path is not None:
        torch.save(
            {
                "train_demos": train_trajs,
                "test_demos": test_trajs,
                "train_demo_stats": train_demo_stats,
                "test_demo_stats": test_demo_stats,
                "downsample_time_step": downsample_time_step,
                "scale": scale,
            },
            f=save_path,
        )

    return train_trajs, test_trajs


def get_r3m_img(r3m_model, img):

    transforms = T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
    )  # ToTensor() divides by 255

    img_preproc = transforms(Image.fromarray(img.astype(np.uint8))).reshape(
        -1, 3, 224, 224
    )
    return r3m_model(img_preproc * 255.0)[0].detach()


def get_grad_cam(r3m_model, forward_model, img_cur, img_next, obs_dict):
    # Setup GradCAM

    model = R3MAndForwardModel(forward_model, copy.deepcopy(r3m_model))
    target_layers = [model.r3m_model.module.convnet.layer4[-1]]
    cam_pred = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    cam_gt = GradCAM(
        model=r3m_model,
        target_layers=[r3m_model.module.convnet.layer4[-1]],
        use_cuda=False,
    )

    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    img_cur_preproc = transforms(Image.fromarray(img_cur.astype(np.uint8))).reshape(
        -1, 3, 224, 224
    )
    img_next_preproc = transforms(Image.fromarray(img_next.astype(np.uint8))).reshape(
        -1, 3, 224, 224
    )

    obs = get_obs_vec_from_obs_dict(obs_dict)

    img_cur_preproc_flat = torch.unsqueeze(img_cur_preproc.flatten(), 0) * 255.0
    # input_tensor = torch.cat([img_cur_preproc_flat, obs], dim=1)
    input_tensor = img_cur_preproc_flat

    grayscale_hm_pred = cam_pred(
        input_tensor=input_tensor, targets=None
    )  # [input_tensor.shape]
    grayscale_hm_gt = cam_gt(input_tensor=img_next_preproc * 255.0, targets=None)

    # grayscale_cam = cam(input_tensor=img_cur_preproc*255.0, targets=None)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_hm_pred = grayscale_hm_pred[0, :]
    grayscale_hm_pred = grayscale_hm_pred[:, : 3 * 224 * 224]
    grayscale_hm_pred = grayscale_hm_pred.reshape(3, 224, 224).transpose(1, 2, 0)
    img_next_postproc = img_next_preproc[0].detach().numpy().transpose(1, 2, 0)
    visualization_pred = show_cam_on_image(
        img_next_postproc, grayscale_hm_pred, use_rgb=True
    )

    grayscale_hm_gt = grayscale_hm_gt[0, :]
    visualization_gt = show_cam_on_image(
        np.float32(img_next_preproc[0]).transpose(1, 2, 0),
        grayscale_hm_gt,
        use_rgb=True,
    )

    return visualization_pred, visualization_gt


def resize_img(img, new_dim=64):

    resize = T.Compose(
        [T.Resize(new_dim, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
    )
    resized_img = resize(Image.fromarray(img.astype(np.uint8)))
    # resized_img = resized_img.detach().numpy().transpose(1,2,0) * 255.0
    return resized_img


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
    elif obj_state_type == "img_r3m":
        o_state = traj["image_60_r3m"][t]
    else:
        raise ValueError

    obs_dict["o_state"] = torch.unsqueeze(torch.FloatTensor(o_state), 0)

    return obs_dict


def parse_pred_traj(pred_traj, state, fnum=3, mpc_use_ftpos=True):
    """
    Parse out relevant part of pred_traj

    args:
        pred_traj (nparray [T, state_dim]):  where each row is a state vector of the format [ftpos, o_state]
        state (str): "ftpos" | "obj" | "ftpos_obj"
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


def save_gif(images, save_str):

    frames = []
    for i, img in enumerate(images):
        # img = resize_img(img).detach().numpy().transpose(1,2,0) * 255.
        frames.append(img.astype(np.uint8))
    imageio.mimsave(save_str, frames)
