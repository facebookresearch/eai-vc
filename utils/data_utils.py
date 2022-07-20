import numpy as np
import matplotlib.pyplot as plt
import json
import os
import torch

NON_TRAJ_KEYS = ["ft_pos_targets_per_mode"]

def get_traj_dict_from_obs_list(data, scale=1, include_image_obs=True):
    """
    Process list of observation dicts into dict of lists (trajectories for each quantity)

    args:
        data: list of observation dicts
        scale: amount to scale all distances by. by default, all distances are in meters. convert to cm with scale=100
    """

    position_error = np.array([data[i]["achieved_goal"]["position_error"] for i in range(len(data))])
    o_cur = np.array([data[i]["object_observation"]["position"] for i in range(len(data))])
    o_des = np.array([data[i]["desired_goal"]["position"] for i in range(len(data))])
    o_cur_ori = np.array([data[i]["object_observation"]["orientation"] for i in range(len(data))])
    o_des_ori = np.array([data[i]["desired_goal"]["orientation"] for i in range(len(data))])
    ft_pos_cur = np.array([data[i]["policy"]["controller"]["ft_pos_cur"] for i in range(len(data))])
    ft_pos_des = np.array([data[i]["policy"]["controller"]["ft_pos_des"] for i in range(len(data))])
    delta_ftpos = np.array([data[i]["action"]["delta_ftpos"] for i in range(len(data))])
    ft_vel_cur = np.array([data[i]["policy"]["controller"]["ft_vel_cur"] for i in range(len(data))])
    ft_vel_des = np.array([data[i]["policy"]["controller"]["ft_vel_des"] for i in range(len(data))])

    t = np.expand_dims(np.array([data[i]["t"] for i in range(len(data))]), 1)
    
    traj_dict = {
                "t"          : t,
                "o_pos_cur"  : scale * o_cur,
                "o_pos_des"  : scale * o_des,
                "o_ori_cur"  : scale * o_cur_ori,
                "o_ori_des"  : scale * o_des_ori,
                "ft_pos_cur" : scale * ft_pos_cur,
                "ft_pos_des" : scale * ft_pos_des,
                "ft_vel_cur" : scale * ft_vel_cur,
                "ft_vel_des" : scale * ft_vel_des,
                "position_error": scale * position_error,
                "delta_ftpos": scale * delta_ftpos,
                }

    if include_image_obs:
        image60 = np.array([data[i]['camera_observation']['camera60']['image'] for i in range(len(data))])
        image180 = np.array([data[i]['camera_observation']['camera180']['image'] for i in range(len(data))])
        image300 = np.array([data[i]['camera_observation']['camera300']['image'] for i in range(len(data))])
        traj_dict["image_60"] = image60
        traj_dict["image_180"] = image180
        traj_dict["image_300"] = image300

    # Mode information
    if "ft_pos_targets_per_mode" in data[-1]["policy"]:
        traj_dict["ft_pos_targets_per_mode"] = scale * data[-1]["policy"]["ft_pos_targets_per_mode"]
        
        # Add "mode" 
        if "mode" not in data[0]["policy"]:
            traj_dict["mode"] = np.array([len(data[i]["policy"]["ft_pos_targets_per_mode"]) \
                                            for i in range(len(data))])
        else:
            traj_dict["mode"] = np.array([data[i]["policy"]["mode"] for i in range(len(data))])
            

    # Object vertices
    if "vertices" in data[0]["object_observation"]:
        vertices = []
        # Flatten vertices dict at each timestep and add to vertices list
        for i in range(len(data)):
            v_wf_dict = data[i]["object_observation"]["vertices"]
            v_wf_flat = np.zeros(len(v_wf_dict) * 3)
            for k, v_wf in v_wf_dict.items():
                v_wf_flat[k*3:k*3+3] = v_wf
            vertices.append(v_wf_flat)

        traj_dict["vertices"] = scale * np.array(vertices)

    return traj_dict

def downsample_traj_dict(traj_dict, cur_time_step=0.004, new_time_step=0.1):
    """ Downsample each of the trajectories in traj_dict """

    every_x_steps = max(1, int(new_time_step / cur_time_step))
    num_waypoints = int(traj_dict["t"].shape[0] / every_x_steps)
    indices_to_take = np.linspace(0, traj_dict["t"].shape[0]-1, num_waypoints, dtype=int)
    
    new_traj_dict = {}
    
    for k, traj in traj_dict.items():
        if "delta" in k: continue # Need to recompute deltas for downsampled traj

        if k in NON_TRAJ_KEYS:
            new_traj_dict[k] = traj
            continue

        #new_traj = traj[::every_x_steps, :]
        new_traj = traj[indices_to_take]
        new_traj_dict[k] = new_traj

    # Compute deltas for downsampled traj
    new_delta_ftpos = np.zeros(new_traj_dict["ft_pos_cur"].shape)
    ft_pos = new_traj_dict["ft_pos_cur"]
    for t in range(ft_pos.shape[0] - 1):
        delta = ft_pos[t+1] - ft_pos[t]
        new_delta_ftpos[t, :] = delta 
    new_traj_dict["delta_ftpos"] = new_delta_ftpos

    return new_traj_dict

def crop_traj_dict(traj_dict, crop_range):
    """ crop_range: [crop_min, crop_max] """


    if crop_range[0] is None: crop_min = 0
    else: crop_min = crop_range[0]

    new_traj_dict = {}
    
    for k, traj in traj_dict.items():
        if crop_range[1] is None: crop_max = traj.shape[0]
        else: crop_max = crop_range[1]

        if k in NON_TRAJ_KEYS:
            new_traj_dict[k] = traj
            continue

        new_traj = traj[crop_min:crop_max, :]
        new_traj_dict[k] = new_traj

    return new_traj_dict
    
def plot_traj(title, save_path, d_list, data_dicts, plot_timestamp = None):
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

            if "x" in data: x = data["x"]
            else: x = list(range(num_steps))

            if "marker" in data: marker = data["marker"]
            else: marker = None

            plt.plot(x, data["y"][:, i], marker=marker, label=label)

        if plot_timestamp is not None:
            plt.axvline(x=plot_timestamp, ls='--', c='k', lw=1)

    plt.legend()


    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

def load_trajs(exp_info, exp_dir=None, scale=1):
    """
    Load train and test trajectories from exp_info
    
    Args
        exp_info (dict or path to .json file): should contain a dict in the following format:
                        {
                            "demo_dir"   : top-level directory containing demos ("demos/"),
                            "difficulty" : difficulty level (1),
                            "train_demos": list of demo ids for training ([0,1]),
                            "test_demos" : list of demo ids for testing ([5]),
                        }
        exp_dir (str): If specified, save demo_info.pth in exp_dir/
    """

    # Load json and get traj filepaths
    if type(exp_info) is dict:
        info = exp_info
    else:
        with open(exp_info, "rb") as f:
            info = json.load(f)

    demo_dir       = info["demo_dir"]
    train_demo_ids = info["train_demos"]
    test_demo_ids  = info["test_demos"]
    diff           = info["difficulty"]

    downsample_time_step = 0.2

    train_trajs = []
    test_trajs = []

    train_demo_stats = []
    test_demo_stats = []

    # Load and downsample train trajectories
    def get_demo_path(demo_dir, diff, demo_id):
        demo_path = os.path.join(demo_dir, f"difficulty-{diff}", f"demo-{demo_id:04d}.npz")
        return demo_path

    # Load and downsample test trajectories
    for demo_id_list, traj_list, stats_list in [[train_demo_ids, train_trajs, train_demo_stats],\
        [test_demo_ids, test_trajs, test_demo_stats]]:

        for demo_id in demo_id_list:
            demo_path = get_demo_path(demo_dir, diff, demo_id)

            demo_stats = {"path": demo_path, "diff": diff, "id": demo_id}
            stats_list.append(demo_stats)

            data = np.load(demo_path, allow_pickle=True)["data"]
            traj_original = get_traj_dict_from_obs_list(data, scale=scale)
    
            # Full trajectory, downsampled
            traj = downsample_traj_dict(traj_original, new_time_step=downsample_time_step)

            traj_list.append(traj)

    print(f"Loaded {len(train_trajs)} training demos")
    print(f"Loaded {len(test_trajs)} test demos")
    
    # Save demo info (train and test demos)
    if exp_dir is not None:
        torch.save({
            'train_demos'         : train_trajs,
            'test_demos'          : test_trajs,
            'train_demo_stats'    : train_demo_stats,
            'test_demo_stats'     : test_demo_stats,
            'downsample_time_step': downsample_time_step,
            'scale'               : scale,
        }, f=f'{exp_dir}/demo_info.pth')

    return train_trajs, test_trajs

def get_obs_dict_from_traj(traj, t, obj_state_type):

    if obj_state_type == "pos":
        o_state = traj["o_pos_cur"][t]
    elif obj_state_type == "vertices":
        o_state = traj["vertices"][t]
    else: 
        raise ValueError

    obs_dict = {
                "ft_state": torch.unsqueeze(torch.FloatTensor(traj["ft_pos_cur"][t]), 0),
                "o_state" : torch.unsqueeze(torch.FloatTensor(o_state), 0),
                "mode"    : traj["mode"][t],
               }

    return obs_dict

def parse_pred_traj(pred_traj, state, fnum=3):
    """ 
    Parse out relevant part of pred_traj
    
    args:
        pred_traj: [T, state_dim] where each row is a state vector of the format [ftpos, o_state]
    """

    ftpos_dim = 3*fnum

    if state == "ftpos":
        return pred_traj[:, :ftpos_dim] # First part of pred_traj
    elif state == "obj":
        assert pred_traj.shape[1] > ftpos_dim, "State does not include object state. Try using mpc_type = two_phase"
        return pred_traj[:, ftpos_dim:] # Last part of pred_traj
    elif state == "ftpos_obj":
        assert pred_traj.shape[1] > ftpos_dim, "State does not include object state. Try using mpc_type = two_phase"
        return pred_traj
    else:
        raise ValueError(f"{state} is invalid state")
    
