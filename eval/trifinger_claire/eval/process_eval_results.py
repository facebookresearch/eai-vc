import os
import sys
import pandas as pd
import argparse
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

"""
Aggregate results from eval_traj_*_log.npz files in exp_*/eval/ directories

Takes a path to a top-level directory containing many exp_*/ directories, and will find all
eval_traj_*_log.npz files in all exp_*/eval/ sub-directories

Example usage:
python scripts/compile_eval_results.py /Users/clairelchen/projects/trifinger_claire/trifinger_mbirl/logs/runs/same_train_and_test 
"""

def load_and_save_df(top_dir):
    data_dict = {
                "algo"                 : [],
                "final_pos_err"        : [],
                "cost_type"            : [],
                "test_loss"            : [],
                "mpc_type"             : [],
                #"n_inner_iter"        : [],
                "action_lr"            : [],
                "cost_lr"              : [],
                "train_or_test"        : [],
                "traj_id"              : [],
                "n_train_traj"         : [],
                "f1_final_dist_to_obj" : [],
                "f2_final_dist_to_obj" : [],
                "f3_final_dist_to_obj" : [],
                }

    # Find all experiment directories in top_dir 
    for item_name in os.listdir(top_dir):
        
        if os.path.isfile(item_name): continue # skip non-directories

        # Check for exp_dir/eval/
        exp_dir = os.path.join(top_dir, item_name)
        top_eval_dir = os.path.join(exp_dir, "eval")
        if not os.path.exists(top_eval_dir): continue # skip directory if it doesn't contain eval/

        # Load configuration, and set parameters based on algo
        conf_file_path = os.path.join(exp_dir, "conf.pth")
        conf = torch.load(conf_file_path)
        algo = conf["algo"]["name"]

        # Load expert demo info
        demo_info_path = os.path.join(exp_dir, "demo_info.pth")
        demo_info = torch.load(demo_info_path)

        # Find latest checkpoint
        ckpts_dir = os.path.join(exp_dir, "ckpts")
        epoch = 0
        for item in os.listdir(ckpts_dir):
            if item.endswith('ckpt.pth'):
                epoch = max(epoch, int(item.split('_')[1]))
        ckpt_name = f"epoch_{epoch}_ckpt.pth"
        ckpt_path = os.path.join(ckpts_dir, ckpt_name)
        ckpt_info = torch.load(ckpt_path)

        # Iterate through test and train eval results
        for train_or_test in ["test", "train"]:
            eval_dir = os.path.join(top_eval_dir, train_or_test)
            n_demos = len(demo_info[f"{train_or_test}_demos"])
            for traj_num in range(n_demos):

                if algo == "mbirl":
                    test_loss = ckpt_info[f"irl_loss_{train_or_test}_per_demo"][traj_num].detach().item()
                else:
                    test_loss = np.nan
                
                log_file_name = f"traj_{traj_num}_log.npz"
                log_path = os.path.join(eval_dir, log_file_name)
                npz_dict = np.load(log_path, allow_pickle=True)
                data = npz_dict["data"]
                demo_data = npz_dict["demo_data"][0]

                final_pos_err = data[-1]["achieved_goal"]["position_error"] # final position error

                # Compute final distance to object center for each fingertip
                ftpos = data[-1]["policy"]["controller"]["ft_pos_cur"]
                obj_pos = data[-1]["object_observation"]["position"]
                for i in range(3):
                    ftpos_i = ftpos[i*3:i*3+3]
                    dist = np.linalg.norm(ftpos_i - obj_pos)
                    data_dict[f"f{i+1}_final_dist_to_obj"].append(dist)

                if algo == "mbirl":
                    irl_loss_state = conf["algo"]["irl_loss_state"]
                    cost_state = conf["algo"]["cost_state"]
                    label = f"{algo}_irl-{irl_loss_state}_cost-{cost_state}"

                    cost_type = conf["algo"]["cost_type"]
                    mpc_type = conf["algo"]["mpc_type"]
                    action_lr = conf["algo"]["action_lr"]
                    cost_lr = conf["algo"]["cost_lr"]
                else:
                    label = algo + "-" + conf["algo"]["obs_type"]

                    cost_type = "na"
                    mpc_type = "na"
                    action_lr = "na"
                    cost_lr = "na"

                data_dict["algo"].append(label)
                data_dict["cost_type"].append(cost_type)
                data_dict["final_pos_err"].append(final_pos_err)
                data_dict["test_loss"].append(test_loss)
                data_dict["train_or_test"].append(train_or_test)
                data_dict["traj_id"].append(demo_data["id"])
                data_dict["n_train_traj"].append(demo_data["n_train_traj"])
                data_dict["mpc_type"].append(mpc_type)
                #data_dict["n_inner_iter"].append(conf.n_inner_iter)
                data_dict["action_lr"].append(action_lr)
                data_dict["cost_lr"].append(cost_lr)

                # ADD ROW FOR GROUND TRUTH DEMOS
                expert_demo = demo_info[f"{train_or_test}_demos"][traj_num]
                scale = demo_info["scale"]
                data_dict["algo"].append("expert")
                data_dict["cost_type"].append("na")
                data_dict["final_pos_err"].append(expert_demo["position_error"][-1]/scale)
                data_dict["test_loss"].append(np.nan)
                data_dict["train_or_test"].append(train_or_test)
                data_dict["traj_id"].append(demo_data["id"])
                data_dict["n_train_traj"].append(np.nan)
                data_dict["mpc_type"].append("na")
                #data_dict["n_inner_iter"].append(conf.n_inner_iter)
                data_dict["action_lr"].append(np.nan)
                data_dict["cost_lr"].append(np.nan)
                # Compute final distance to object center for each fingertip
                ftpos = expert_demo["ft_pos_cur"][-1]/scale
                obj_pos = expert_demo["o_pos_cur"][-1]/scale
                for i in range(3):
                    ftpos_i = ftpos[i*3:i*3+3]
                    dist = np.linalg.norm(ftpos_i - obj_pos)
                    data_dict[f"f{i+1}_final_dist_to_obj"].append(dist)


    df = pd.DataFrame.from_dict(data_dict) 

    csv_path = os.path.join(top_dir, "eval_results.csv")
    df.to_csv(csv_path)
    
    return df

def main(args):
    
    top_dir = args.top_dir

    csv_file_name = os.path.join(top_dir, "eval_results.csv")
    if os.path.exists(csv_file_name):
        df = pd.read_csv(csv_file_name)
    else:
        df = load_and_save_df(top_dir)

    #df = df.loc[(df["cost_lr"] == 1e-3) & (df["action_lr"] == 1e-2)] # filter for  multirun/2022-08-01/17-39-55 
    df = df.loc[(df["mpc_type"] != "ftpos_obj_two_phase")] # for obj state = pos

### PLOTTING

    all_algos = [
            "mbirl_irl-ftpos_obj_cost-ftpos_obj",
            "mbirl_irl-ftpos_cost-ftpos_obj",
            "mbirl_irl-obj_cost-ftpos_obj",
            "mbirl_irl-ftpos_obj_cost-ftpos",
            "mbirl_irl-ftpos_cost-ftpos",
            "mbirl_irl-obj_cost-ftpos",
            "mbirl_irl-ftpos_obj_cost-obj",
            "mbirl_irl-ftpos_cost-obj",
            "mbirl_irl-obj_cost-obj",
            "bc-goal_rel",
            "bc-img_r3m",
            "expert",
            ]


    label_dict = {
                "mbirl_irl-ftpos_obj_cost-ftpos_obj": "mbirl\nirl loss: ftpos_obj\ncost: ftpos_obj",
                "mbirl_irl-ftpos_obj_cost-ftpos": "mbirl\nirl loss: ftpos_obj\ncost: ftpos",
                "mbirl_irl-ftpos_obj_cost-obj": "mbirl\nirl loss: ftpos_obj\ncost: obj",
                "mbirl_irl-ftpos_cost-ftpos_obj": "mbirl\nirl loss: ftpos\ncost: ftpos_obj",
                "mbirl_irl-ftpos_cost-ftpos": "mbirl\nirl loss: ftpos\ncost: ftpos",
                "mbirl_irl-ftpos_cost-obj": "mbirl\nirl loss: ftpos\ncost: obj",
                "mbirl_irl-obj_cost-ftpos_obj": "mbirl\nirl loss: obj\ncost: ftpos_obj",
                "mbirl_irl-obj_cost-ftpos": "mbirl\nirl loss: obj\ncost: ftpos",
                "mbirl_irl-obj_cost-obj": "mbirl\nirl loss: obj\ncost: obj",
                "bc-goal_rel": "bc\ngoal_rel",
                "bc-img_r3m": "bc\nimg_r3m",
                "expert": "expert"
                }

    order = [algo for algo in all_algos if algo in df["algo"].tolist()]
    x_tick_labels = [label_dict[algo] for algo in order]
    
    all_hue_order = ["Traj", "MPTimeDep", "TimeDep", "Weighted", "na"]
    hue_order = [c for c in all_hue_order if c in df["cost_type"].tolist()]


    ## BAR PLOTS
    #plt.figure(figsize=(15, 10), dpi=200)
    #ax = sns.barplot(x="algo", y="final_pos_err", hue="cost_type", data=df, palette="Set3", hue_order=hue_order, order=order)
    #plt.ylabel("Final object position error (m)")
    #plt.xlabel("Method")
    #ax.set_xticklabels(x_tick_labels)
    ##plt.title("Final object position error")
    #plt.legend(title="Cost type")
    #if args.save:
    #    save_path = os.path.join(top_dir, "final_err.png")
    #    plt.savefig(save_path)
    #else:
    #    plt.show()
    #plt.figure(figsize=(15, 10), dpi=200)
    #ax = sns.barplot(x="algo", y="test_loss", hue="cost_type", data=df, palette="Set3", hue_order=hue_order, order=order)
    #plt.ylabel("Final loss")
    #plt.xlabel("Method")
    #ax.set_xticklabels(x_tick_labels)
    #plt.title("Final IRL loss on test traj")
    #plt.legend(title="Cost type")
    #if args.save:
    #    save_path = os.path.join(top_dir, "test_loss.png")
    #    plt.savefig(save_path)
    #else:
    #    plt.show()

    # Plot mpc_type==ftpos_obj_two_phase vs mpc_type==ftpos_obj_learned_only
    #df = df.loc[(df["mpc_type"] != "na") & (df["train_or_test"] == "test")]
    #row_cat = "mpc_type"
    
    row_cat = "train_or_test"

    ## CAT PLOT
    plt.figure()
    ax = sns.catplot(x="algo", y="final_pos_err", hue="cost_type", data=df, palette="Set3", hue_order=hue_order, order=order, 
                    kind="bar", row=row_cat, height=6, aspect=2.5, sharex=False, sharey=True, legend=False)
    ax.set_axis_labels("Method", "Final position error (m)")
    ax.set_xticklabels(x_tick_labels)
    ax.set_titles("{row_name} goals")
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Final object position error")
    plt.tight_layout()
    plt.legend(loc='upper right')
    if args.save:
        save_path = os.path.join(top_dir, "final_err.png")
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

    for i in range(1,4):
        plt.figure()
        ax = sns.catplot(x="algo", y=f"f{i}_final_dist_to_obj", hue="cost_type", data=df, palette="Set3", hue_order=hue_order, order=order, 
                        kind="bar", row=row_cat, height=6, aspect=2.5, sharex=False, sharey=True, legend=False)
        ax.set_axis_labels("Method", "Final distance to obj (m)")
        ax.set_xticklabels(x_tick_labels)
        ax.set_titles("{row_name} goals")
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Finger {i} final distance to obj")
        plt.tight_layout()
        plt.legend(loc='upper right')
        if args.save:
            save_path = os.path.join(top_dir, f"f{i}_final_dist_to_obj.png")
            plt.savefig(save_path, dpi=200)
        else:
            plt.show()

    plt.figure()
    ax = sns.catplot(x="algo", y="test_loss", hue="cost_type", data=df, palette="Set3", hue_order=hue_order, order=order, 
                    kind="bar", row=row_cat, height=6, aspect=2.5, sharex=False, sharey=True, legend=False)
    ax.set_axis_labels("Method", "Final loss")
    ax.set_xticklabels(x_tick_labels)
    ax.set_titles("{row_name} goals")
    plt.suptitle("Final IRL loss")
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    plt.legend(loc='upper right')
    if args.save:
        save_path = os.path.join(top_dir, "final_loss.png")
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("top_dir", default=None, help="""Filepath of top-level directory containing experiment directories""")
    parser.add_argument("--save", "-s", action="store_true")
    args = parser.parse_args()
    main(args)



