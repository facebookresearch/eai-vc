import numpy as np
import matplotlib.pyplot as plt

NON_TRAJ_KEYS = ["ft_pos_targets_per_mode"]

def get_traj_dict_from_obs_list(data, scale=1):

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
                "o_cur_pos"  : scale * o_cur,
                "o_des_pos"  : scale * o_des,
                "o_cur_ori"  : scale * o_cur_ori,
                "o_des_ori"  : scale * o_des_ori,
                "ft_pos_cur" : scale * ft_pos_cur,
                "ft_pos_des" : scale * ft_pos_des,
                "ft_vel_cur" : scale * ft_vel_cur,
                "ft_vel_des" : scale * ft_vel_des,
                "delta_ftpos": scale * delta_ftpos,
                }

    if "ft_pos_targets_per_mode" in data[-1]["policy"]:
        traj_dict["ft_pos_targets_per_mode"] = scale * data[-1]["policy"]["ft_pos_targets_per_mode"]

    return traj_dict

def downsample_traj_dict(traj_dict, cur_time_step=0.004, new_time_step=0.1):
    """ Downsample each of the trajectories in traj_dict """

    every_x_steps = max(1, int(new_time_step / cur_time_step))
    
    new_traj_dict = {}
    
    for k, traj in traj_dict.items():
        if "delta" in k: continue # Need to recompute deltas for downsampled traj

        if k in NON_TRAJ_KEYS:
            new_traj_dict[k] = traj
            continue

        new_traj = traj[::every_x_steps, :]
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

    plt.legend()

    if plot_timestamp is not None:
        plt.axvline(x=plot_timestamp, ls='--', c='k', lw=1)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

