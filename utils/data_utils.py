import numpy as np

def get_traj_dict_from_obs_list(data):

    position_error = np.array([data[i]["achieved_goal"]["position_error"] for i in range(len(data))])
    o_cur = np.array([data[i]["object_observation"]["position"] for i in range(len(data))])
    o_des = np.array([data[i]["desired_goal"]["position"] for i in range(len(data))])
    ft_pos_cur = np.array([data[i]["policy"]["controller"]["ft_pos_cur"] for i in range(len(data))])
    ft_pos_des = np.array([data[i]["policy"]["controller"]["ft_pos_des"] for i in range(len(data))])
    delta_ftpos = np.array([data[i]["action"]["delta_ftpos"] for i in range(len(data))])

    traj_dict = {
                "o_cur"      : o_cur,
                "o_des"      : o_des,
                "ft_pos_cur" : ft_pos_cur,
                "ft_pos_des" : ft_pos_cur,
                "delta_ftpos": delta_ftpos,
                }

    return traj_dict
