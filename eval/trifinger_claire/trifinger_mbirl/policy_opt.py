# Copyright (c) Facebook, Inc. and its affiliates.
import random
import os
import sys
import torch
import numpy as np
import higher
import matplotlib.pyplot as plt
import wandb
import logging

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

import utils.data_utils as d_utils
import utils.train_utils as t_utils

from trifinger_mbirl.ftpos_mpc import FTPosMPC
from trifinger_mbirl.two_phase_mpc import TwoPhaseMPC
from trifinger_mbirl.learned_mpc import LearnedMPC
from trifinger_mbirl.forward_models.models.decoder_model import DecoderModel
from trifinger_mbirl.sim_mpc import SimMPC
from trifinger_mbirl.mbirl import get_expert_demo 

# A logger for this file
log = logging.getLogger(__name__)

class MSETargetCost(object):

    def __call__(self, y_in, y_target):
        assert y_in.dim() == 2
        sqrd_error = ((y_in - y_target) ** 2) # [time_horizon, dim]
        return sqrd_error.mean()

class PolicyOpt:
    def __init__(self, conf, traj_info, device):

        time_horizon = traj_info["train_demos"][0]["ft_pos_cur"].shape[0]
        
        self.conf = conf
        self.device = device
        self.traj_info = traj_info
        self.downsample_time_step = traj_info["downsample_time_step"]

        # Get MPC
        self.mpc = get_mpc(conf.mpc_type, time_horizon, self.device, conf.mpc_forward_model_ckpt)
        log.info(f"Loaded MPC type {self.conf.mpc_type} with obj_state_type = {self.mpc.obj_state_type}")

        # Ensure mpc_type and cost state are compatible
        if conf.mpc_type == "learned" and ("ftpos" in conf.cost_state):
            assert mpc.use_ftpos, "Forward model does not use ftpos, but cost_state requires ftpos"

        self.mpc_use_ftpos = True
        if conf.mpc_type == "learned" and not self.mpc.use_ftpos: self.mpc_use_ftpos = False
    
        # Get learnable cost function
        self.cost = MSETargetCost()

        # Load and use decoder to viz pred_o_states
        if conf.path_to_decoder_ckpt is not None:
            decoder_model_dict = torch.load(conf.path_to_decoder_ckpt, map_location=torch.device(self.device))
            self.decoder = DecoderModel()
            self.decoder.load_state_dict(decoder_model_dict["model_state_dict"])
            self.decoder.to(self.device)
        else:
            self.decoder = None

        self.sim = SimMPC(downsample_time_step=self.downsample_time_step, traj_scale=self.traj_info["scale"])

    def train(self, model_data_dir=None, no_wandb=False):

        torch.autograd.set_detect_anomaly(True)
        train_trajs = self.traj_info["train_demos"]
        test_trajs = self.traj_info["test_demos"]

        # Make logging directories
        ckpts_dir = os.path.join(model_data_dir, "ckpts") 
        plots_dir = os.path.join(model_data_dir, "train")
        test_plots_dir = os.path.join(model_data_dir, "test")
        if not os.path.exists(ckpts_dir): os.makedirs(ckpts_dir)
        if not os.path.exists(plots_dir): os.makedirs(plots_dir)
        if not os.path.exists(test_plots_dir): os.makedirs(test_plots_dir)

        for demo_i in range(len(train_trajs)):
            expert_demo_dict = train_trajs[demo_i]

            # Init state
            obs_dict_init = d_utils.get_obs_dict_from_traj(expert_demo_dict, 0, self.mpc.obj_state_type) # Initial state

            # Reset mpc for action optimization
            expert_actions = torch.Tensor(expert_demo_dict["delta_ftpos"][:-1]).to(self.device)
            self.mpc.reset_actions()
            #self.mpc.reset_actions(init_a=expert_actions)

            #action_optimizer = torch.optim.SGD(self.mpc.parameters(), lr=self.conf.action_lr)
            action_optimizer = torch.optim.Adam(self.mpc.parameters(), lr=self.conf.action_lr)
            action_optimizer.zero_grad()

            for inner_i in range(self.conf.n_inner_iter):

                print(f"Iter {inner_i}")
                pred_traj = self.mpc.roll_out(obs_dict_init.copy())

                #for n, p in self.mpc.named_parameters():
                #    print(n, p.grad)
                #    if p.grad is not None:
                #        grad = p.grad.detach()
                #        #print(name, grad.shape)
                #        grad_min_mpc = torch.min(torch.abs(grad))
                #        grad_max_mpc = torch.max(torch.abs(grad))
                #        grad_norm_mpc = grad.norm()
                #        print(grad_norm_mpc, grad_min_mpc, grad_max_mpc)

                #pred_actions = self.mpc.action_seq.detach().numpy()
                #pred_traj_sim = torch.Tensor(self.sim.rollout_actions(expert_demo_dict, pred_actions))

                target = self.get_target_for_cost_type(expert_demo_dict)
                cost_val = self.cost(pred_traj, target)
                print("loss: ", str(cost_val.item()))
                cost_val.backward()
                action_optimizer.step()

                if (inner_i+1) % self.conf.n_epoch_every_log == 0:
                    diff = self.traj_info["train_demo_stats"][demo_i]["diff"]
                    traj_i = self.traj_info["train_demo_stats"][demo_i]["id"]
                    traj_plots_dir = os.path.join(plots_dir, f"diff-{diff}_traj-{traj_i}")
                    if not os.path.exists(traj_plots_dir): os.makedirs(traj_plots_dir)
                    pred_actions = self.mpc.action_seq.clone().data.cpu().detach().numpy()
                    self.plot(traj_plots_dir, inner_i, pred_traj, pred_actions, expert_demo_dict)
                    
                #    torch.save({
                #        'train_pred_traj_per_demo'   : pred_traj.detach(), 
                #        'train_pred_actions_per_demo': pred_actions,
                #        'conf'                       : self.conf,
                #    }, f=f'{ckpts_dir}/epoch_{outer_i*self.conf.n_inner_iter+inner_i+1}_ckpt.pth')
            #################### End inner loop: policy optimization ############################

    def plot(self, traj_plots_dir, outer_i, pred_traj, pred_actions, expert_demo_dict):

        if self.mpc_use_ftpos:
            plot_traj(traj_plots_dir, outer_i, pred_traj, expert_demo_dict, self.conf.mpc_type, self.mpc.obj_state_type)

        if self.conf.mpc_type=="learned" and self.decoder is not None:
            ft_states, o_states = self.mpc.get_states_from_x_next(pred_traj)
            with torch.no_grad(): # TODO need to freeze decoder weights??
                pred_imgs = self.decoder(torch.Tensor(o_states))
                self.decoder.save_gif(pred_imgs, os.path.join(traj_plots_dir, f'r3m_epoch_{outer_i+1}.gif'))

        plot_actions(traj_plots_dir, outer_i, pred_actions, expert_demo_dict)

    def get_target_for_cost_type(self, demo_dict):
        """ Get target from traj_dict for learnable cost function given cost_type  and cost_state """

        cost_state = self.conf.cost_state
        obj_state_type = self.mpc.obj_state_type

        expert_demo = get_expert_demo(demo_dict, cost_state, obj_state_type)

        target = expert_demo[-1]

        return target.to(self.device)

def plot_traj(plots_dir, outer_i, pred_traj, expert_demo_dict, mpc_type, obj_state_type):
    """ Plot predicted and expert trajectories, based on mpc_type """

    title = "Predicted trajectories (outer i: {})".format(outer_i)
    save_name = f"traj_epoch_{outer_i+1}.png"
    save_path = os.path.join(plots_dir, save_name)

    if "ftpos_obj" in mpc_type and obj_state_type == "pos": # TODO hardcoded
        d_list = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "ox", "oy", "oz"]
        plot_expert_demo = get_expert_demo(expert_demo_dict, "ftpos_obj", obj_state_type)
    else:
        d_list = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3"]
        plot_expert_demo = get_expert_demo(expert_demo_dict, "ftpos", obj_state_type)

    d_utils.plot_traj(
            title, 
            save_path,
            d_list,
            {
            "pred":  {"y": pred_traj.detach().numpy(), "x": expert_demo_dict["t"], "marker": "x"},
            "demo":  {"y": plot_expert_demo.detach().numpy(), "x": expert_demo_dict["t"], "marker": "."},
            }
            )

def plot_actions(plots_dir, outer_i, pred_actions, expert_demo_dict):
    """ Plot predicted and expert actions """

    expert_actions = expert_demo_dict["delta_ftpos"]
    title = "Fingertip position deltas (outer i: {})".format(outer_i)
    save_name = f"action_epoch_{outer_i+1}.png"
    save_path = os.path.join(plots_dir, save_name)

    d_utils.plot_traj(
            title, 
            save_path,
            ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
            {
            "pred":  {"y": pred_actions, "x": expert_demo_dict["t"][:-1], "marker": "x"},
            "demo":  {"y": expert_actions[:-1], "x": expert_demo_dict["t"][:-1],"marker": "."},
            }
            )

def get_mpc(mpc_type, time_horizon, device, mpc_forward_model_ckpt=None):
    """ Get MPC class """

    if mpc_type == "ftpos":
        return FTPosMPC(time_horizon=time_horizon-1)

    elif mpc_type == "two_phase":
        if mpc_forward_model_ckpt is None: raise ValueError("Missing mpc_forward_model_ckpt")

        ## Phase 2 model trained with cropped phase 2 demos
        #phase2_model_path = "/Users/clairelchen/projects/trifinger_claire/trifinger_mbirl/forward_models/runs/phase2_model_nt-100_ost-pos_train-m2/epoch_1500_ckpt.pth"
        ## Phase 2 model trained with full demos
        #phase2_model_path = "trifinger_mbirl/forward_models/runs/phase2_model_nt-100_ost-pos_train-all/epoch_3000_ckpt.pth"

        return TwoPhaseMPC(time_horizon-1, mpc_forward_model_ckpt)

    elif mpc_type == "learned":
        model_dict = torch.load(mpc_forward_model_ckpt, map_location=torch.device(device))
        return LearnedMPC(time_horizon-1, model_dict=model_dict, device=device).to(device)

    else:
        raise ValueError(f"{mpc_type} is invalid mpc_type")

