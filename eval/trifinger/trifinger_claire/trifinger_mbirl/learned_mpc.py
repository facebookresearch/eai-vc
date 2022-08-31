import torch
import numpy as np
import argparse
import os
import sys
import cv2
import imageio

from r3m import load_r3m

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, ".."))

from trifinger_mbirl.forward_models.models.forward_model import ForwardModel, get_obs_vec_from_obs_dict
from trifinger_mbirl.forward_models.train_forward_model import ForwardModelTrainer
from trifinger_mbirl.forward_models.models.decoder_model import DecoderModel
import utils.data_utils as d_utils
from trifinger_mbirl.policy import DeterministicPolicy

# Compute next state given current state and action (ft position deltas)
class LearnedMPC(torch.nn.Module):

    def __init__(self, time_horizon, model_dict=None, f_num=3, device="cpu"):
        super().__init__()
        self.time_horizon = time_horizon
        self.f_num = f_num
        self.f_state_dim = self.f_num * 3
        self.a_dim = self.f_num * 3
        self.policy_type = "actions"
        self.device = device

        self.in_dim = model_dict["in_dim"]
        self.out_dim = model_dict["out_dim"]
        hidden_dims = model_dict["hidden_dims"]
        self.model = ForwardModel(self.in_dim, self.out_dim, hidden_dims)
        self.model.load_state_dict(model_dict["model_state_dict"])
        self.model.to(device)

        self.forward_model_trainer = ForwardModelTrainer(model=self.model, device=device, conf=model_dict["conf"])

        if self.policy_type == "actions":
            self.action_seq = torch.nn.Parameter(torch.Tensor(np.zeros([time_horizon, self.a_dim])))
        elif self.policy_type == "nn":
            self.policy = DeterministicPolicy(in_dim=self.out_dim, out_dim=self.a_dim, device=device)
            self.action_seq = torch.Tensor(np.zeros([time_horizon, self.a_dim]))
        else:
            raise ValueError("Invalid policy_type.")

        self.max_a = 2.0 # cm

        # Freeze network params
        self.freeze_forward_model()

        self.obj_state_type = model_dict["conf"]["algo"]["obj_state_type"]
        self.use_ftpos = model_dict["conf"]["algo"]["use_ftpos"]

    def forward(self, obs_dict, action=None):
        """
        Given current state and action, and mode, compute next state

        args:
            obs_dict = {
                        "ft_state": ft positions,
                        "o_state": object state,
                       }
            action

        return:
        """

        if action is None:
            if self.use_ftpos:
                return torch.cat([obs_dict["ft_state"], obs_dict["o_state"]], dim=1).to(self.device)
            else:
                return obs_dict["o_state"].to(self.device)
        else:
            obs_dict["action"] = torch.unsqueeze(action, 0).to(self.device)

        obs = get_obs_vec_from_obs_dict(obs_dict, use_ftpos=self.use_ftpos, device=self.device)

        x_next = self.model(obs)

        return x_next

    def get_states_from_x_next(self, x_next):

        if self.use_ftpos:
            ft_state = x_next[:, : self.f_state_dim]
            o_state = x_next[:, self.f_state_dim :]
        else:
            ft_state = None
            o_state = x_next

        return ft_state, o_state

    def roll_out(self, obs_dict_init):
        """ Given intial state, compute trajectory of length self.time_horizon with actions self.action_seq """
        # Clip actions
        #self.action_seq.data.clamp_(-self.max_a, self.max_a)

        pred_traj = []
        x_next = self.forward(obs_dict_init)

        ft_state, o_state = self.get_states_from_x_next(x_next)

        obs_dict_next = {
            "ft_state": ft_state,
            "o_state": o_state,
        }
        pred_traj.append(torch.squeeze(x_next.clone()))

        for t in range(self.time_horizon):
            if self.policy_type == "actions":
                a = self.action_seq[t]
            elif self.policy_type == "nn":
                a = self.policy(x_next.detach())[0]
                # Clip actions
                #print("before a ", a)
                a = torch.where(a > self.max_a, torch.tensor([self.max_a]), a)
                a = torch.where(a < -self.max_a, -torch.tensor([self.max_a]), a)
                #print("after a ", a)
                self.action_seq[t, :] = a.detach()
            else:
                raise ValueError("Invalid policy_type.")


            #print(obs_dict_next)
            x_next = self.forward(obs_dict_next, a)
            x_next = self.clip(x_next)
            #print(x_next)

            pred_traj.append(torch.squeeze(x_next.clone()))

            ft_state, o_state = self.get_states_from_x_next(x_next)

            obs_dict_next = {
                "ft_state": ft_state,
                "o_state": o_state,
            }

        pred_traj = torch.stack(pred_traj)
        return pred_traj

    def roll_out_gt_state(self, expert_traj):
        """[FOR TESTING] Apply action to ground truth state at each timestep (don't use predicted next state as new initial state)"""
        pred_traj = []

        obs_dict = d_utils.get_obs_dict_from_traj(expert_traj, 0, self.obj_state_type)
        x_next = self.forward(obs_dict)
        pred_traj.append(torch.squeeze(x_next.clone()))

        for t in range(self.time_horizon):
            obs_dict = d_utils.get_obs_dict_from_traj(
                expert_traj, t, self.obj_state_type
            )
            a = self.action_seq[t]
            x_next = self.forward(obs_dict, a)
            x_next = self.clip(x_next)

            pred_traj.append(torch.squeeze(x_next.clone()))

        pred_traj = torch.stack(pred_traj)
        return pred_traj

    def reset_actions(self, init_a=None):
        if self.policy_type == "actions":
            if init_a is None:
                self.action_seq.data = torch.Tensor(np.zeros([self.time_horizon, self.a_dim])).to(self.device)
                # Random actions between [-1., 1.]
                #self.action_seq.data = torch.rand((self.time_horizon, self.a_dim)) * 2. - 1.
            else:
                rand = torch.rand((self.time_horizon, self.a_dim)) * 2.0 - 1.0
                self.action_seq.data = torch.Tensor(init_a)# + rand
        elif self.policy_type == "nn":
            self.policy.reset()
        else:
            raise ValueError("Invalid policy_type.")

    def clip(self, x):
        # TODO hardcoded ranges
        x_min = [-20] * self.out_dim
        x_max = [20] * self.out_dim

        x = torch.Tensor(x).to(self.device)
        x_min = torch.unsqueeze(torch.Tensor(x_min), 0).to(self.device)
        x_max = torch.unsqueeze(torch.Tensor(x_max), 0).to(self.device)

        x = torch.where(x > x_max, x_max, x)
        x = torch.where(x < x_min, x_min, x)

        return x

    def set_action_seq_for_testing(self, action_seq):
        self.action_seq.data = torch.Tensor(action_seq)

    def train_forward_model(self, new_traj_list, n_epochs, model_data_dir, no_wandb=True):
        self.forward_model_trainer.update_data(new_traj_list)
        self.forward_model_trainer.model.reset()
        self.forward_model_trainer.train(n_epochs, model_data_dir, no_wandb=no_wandb)

    def freeze_forward_model(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False


#########################################################################################
############################### MPC EVAL FUNCTIONS ######################################
#########################################################################################


def test_mpc(mpc, traj, epoch, save_dir=None, one_step=False):
    """
    Test MPC with learned model on trajectory.
    Plots L2 norm between predicted and ground truth trajectory
    """

    # Run MPC rollout
    obs_dict_init = d_utils.get_obs_dict_from_traj(traj, 0, mpc.obj_state_type)

    # Run roll_out to get trajectory from initial state
    mpc.set_action_seq_for_testing(traj["delta_ftpos"])

    if one_step:
        pred_traj = mpc.roll_out_gt_state(traj)
    else:
        pred_traj = mpc.roll_out(obs_dict_init)

    pred_traj = pred_traj.cpu().detach().numpy()
    pred_ft_states, pred_o_states = mpc.get_states_from_x_next(pred_traj)
    pred_traj_dict = {"ft_state": pred_ft_states, "o_state": pred_o_states}

    pred_err_dict = get_pred_traj_err(
        pred_traj_dict,
        traj,
        mpc.obj_state_type,
        epoch,
        save_dir=save_dir,
        one_step=one_step,
    )

    return pred_traj_dict, pred_err_dict


def get_pred_traj_err(
    pred_traj, gt_traj, obj_state_type, epoch, save_dir=None, one_step=False
):
    """
    Compute and plot L2 norm between predicted and ground truth trajectory
    """

    save_label = "pred_one_step" if one_step else "pred_rollout"
    time_horizon = gt_traj["ft_pos_cur"].shape[0] - 1

    if obj_state_type == "pos":
        true_o_state = gt_traj["o_pos_cur"]
    elif obj_state_type == "vertices":
        true_o_state = gt_traj["vertices"]
    elif obj_state_type == "img_r3m":
        true_o_state = gt_traj["image_60_r3m"]
    else:
        raise ValueError()

    pred_ft_states = pred_traj["ft_state"]
    pred_o_states = pred_traj["o_state"]

    # Compute L2 distance between each ft and object state
    ft_pos_err = np.ones((time_horizon + 1, 3)) * np.nan
    if pred_ft_states is not None:
        for i in range(3):
            per_finger_err = np.linalg.norm(
                (
                    pred_ft_states[:, i * 3 : i * 3 + 3]
                    - gt_traj["ft_pos_cur"][:, i * 3 : i * 3 + 3]
                ),
                axis=1,
            )
            ft_pos_err[:, i] = per_finger_err

        if save_dir is not None:
            # Plot L2 distance between each predicted ftpos and gt ftpos
            save_str = os.path.join(
                save_dir, f"{save_label}_ftpos_l2_dist_epoch_{epoch}.png"
            )
            d_utils.plot_traj(
                "ft position error (cm)",
                save_str,
                ["f1", "f2", "f3"],
                {
                    "err": {"y": ft_pos_err, "x": gt_traj["t"], "marker": "x"},
                },
            )

    # Compute and plot L2 distance between predicted and gt object states
    if save_dir is not None:
        save_str = os.path.join(
            save_dir, f"{save_label}_o_state_l2_dist_epoch_{epoch}.png"
        )
    else:
        save_str = None

    if obj_state_type == "pos":
        o_state_err = np.expand_dims(
            np.linalg.norm((pred_o_states - true_o_state), axis=1), 1
        )

        d_utils.plot_traj(
            "object position error (cm)",
            save_str,
            ["pos"],
            {
                "err": {"y": o_state_err, "x": gt_traj["t"], "marker": "x"},
            },
        )

    elif obj_state_type == "vertices":
        o_state_err = np.zeros((time_horizon + 1, 8))
        for i in range(8):
            per_vert_err = np.linalg.norm(
                (
                    pred_o_states[:, i * 3 : i * 3 + 3]
                    - true_o_state[:, i * 3 : i * 3 + 3]
                ),
                axis=1,
            )
            o_state_err[:, i] = per_vert_err

        d_utils.plot_traj(
            f"object vertex position error (cm)",
            save_str,
            ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
            {
                "err": {"y": o_state_err, "x": gt_traj["t"], "marker": "x"},
            },
        )

    elif obj_state_type == "img_r3m":
        o_state_err = np.expand_dims(
            np.linalg.norm((pred_o_states - true_o_state), axis=1), 1
        )

        d_utils.plot_traj(
            "R3M embedding L2 distance",
            save_str,
            ["r3m"],
            {
                "err": {"y": o_state_err, "x": gt_traj["t"], "marker": "x"},
            },
        )

    else:
        raise ValueError()

    pred_err_dict = {
        "max_ft_pos_l2_dist": np.max(ft_pos_err),
        "max_o_state_l2_dist": np.max(o_state_err),
        "avg_ft_pos_l2_dist": np.mean(ft_pos_err),
        "avg_o_state_l2_dist": np.mean(o_state_err),
    }

    ## Compute error between each state and first state [FOR TESTING ZERO-ACTIONS]
    # o_state_init = np.tile(pred_o_states[0, :], (time_horizon+1, 1))
    # print(o_state_init.shape)
    # if save_dir is not None:
    #    save_str = os.path.join(save_dir, f"{save_label}_zero_action_o_state_l2_from_init_epoch_{epoch}.png")
    # else:
    #    save_str = None
    # o_state_err = np.expand_dims(np.linalg.norm((pred_o_states - o_state_init), axis=1), 1)
    # d_utils.plot_traj(
    #        "R3M embedding L2 distance",
    #        save_str,
    #        ["r3m"],
    #        {
    #        "err":  {"y": o_state_err, "x": gt_traj["t"], "marker": "x"},
    #        },
    #        )

    return pred_err_dict


def main(args):

    # Load model dict
    model_dict = torch.load(args.model_path)

    demo_path = model_dict["conf"].demo_path  # Get demo path used to train model
    epoch = os.path.basename(args.model_path).split("_")[
        1
    ]  # TODO get epoch from ckpt path

    # Directory to save plots (same directory used to save plots during training)
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(args.model_path)), "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Load train and test trajectories
    traj_info = torch.load(demo_path)

    # Make MPC with trained forward model
    time_horizon = traj_info["train_demos"][0]["ft_pos_cur"].shape[0]
    mpc = LearnedMPC(time_horizon - 1, model_dict=model_dict)

    # Load and use decoder to viz pred_o_states
    model_dict = torch.load(args.decoder_model_path)
    decoder = DecoderModel()
    decoder.load_state_dict(model_dict["model_state_dict"])

    # Iterate through training and test demos
    for split_name in ["train", "test"]:
        for i, traj in enumerate(traj_info[f"{split_name}_demos"]):

            # TODO need to update directory naming
            # diff = traj_info[f"{split_name}_demo_stats"][i]["diff"]
            # traj_i = traj_info[f"{split_name}_demo_stats"][i]["id"]
            # traj_plots_dir = os.path.join(plots_dir, split_name, f"diff-{diff}_traj-{traj_i}")
            traj_plots_dir = os.path.join(plots_dir, f"{split_name}_traj_{i}")
            print(traj_plots_dir)

            # Test full rollout and one-step predictions
            for one_step in [False, True]:
                pred_traj_dict, pred_err_dict = test_mpc(
                    mpc, traj, epoch, one_step=one_step
                )
                pred_imgs = decoder(torch.Tensor(pred_traj_dict["o_state"]))

                save_label = "pred_one_step" if one_step else "pred_rollout"
                decoder.save_gif(
                    pred_imgs,
                    os.path.join(traj_plots_dir, f"{save_label}_epoch_{epoch}.gif"),
                )

    ########################################################################################################
    ## Pass r3m images through grad cam - this didn't work.
    # if mpc.obj_state_type == "img_r3m":
    #    # Load R3M
    #    if torch.cuda.is_available():
    #        device = "cuda"
    #    else:
    #        device = "cpu"
    #    r3m_model = load_r3m("resnet50")  # resnet18, resnet34
    #    r3m_model.eval()
    #    r3m_model.to(device)

    #    obs_dict_init = d_utils.get_obs_dict_from_traj(traj, 0, mpc.obj_state_type)
    #    obs_dict_init["action"] = torch.unsqueeze(torch.FloatTensor(traj["delta_ftpos"][0]), dim=0)
    #    #test image
    #    img_cur = traj["image_60"][0]
    #    img_next = traj["image_60"][1]
    #    gradcam_viz_pred, gradcam_viz_gt = d_utils.get_grad_cam(r3m_model, mpc.model, img_cur, img_next, obs_dict_init)

    #    save_str = os.path.join(plots_dir, "gradcam_pred.png")
    #    cv2.imwrite(save_str, gradcam_viz_pred)

    #    save_str = os.path.join(plots_dir, "gradcam_gt.png")
    #    cv2.imwrite(save_str, gradcam_viz_gt)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", "-m", default=None, help="""Filepath of model ckpt to load"""
    )
    parser.add_argument(
        "--decoder_model_path",
        "-d",
        default=None,
        help="""Filepath of model ckpt to load""",
    )
    args = parser.parse_args()

    main(args)
