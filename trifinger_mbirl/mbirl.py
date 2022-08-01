# Copyright (c) Facebook, Inc. and its affiliates.
import random
import os
import sys
import torch
import numpy as np
import higher
import matplotlib.pyplot as plt
import argparse
import collections
import wandb

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

import utils.data_utils as d_utils

from trifinger_mbirl.learnable_costs import *
from trifinger_mbirl.ftpos_mpc import FTPosMPC
from trifinger_mbirl.two_phase_mpc import TwoPhaseMPC
from trifinger_mbirl.dynamics_models import FTPosSim

# The IRL Loss, the learning objective for the learnable cost functions.
# Measures the distance between the demonstrated fingertip position trajectory and predicted trajectory
class IRLLoss(object):
    def __init__(self, irl_loss_state, obj_state_type):
        self.irl_loss_state = irl_loss_state
        self.obj_state_type = obj_state_type

    def __call__(self, full_pred_traj, expert_demo_dict):
        pred_traj = d_utils.parse_pred_traj(full_pred_traj, self.irl_loss_state)
        target_traj = get_expert_demo(expert_demo_dict, self.irl_loss_state, self.obj_state_type)
        loss = ((pred_traj - target_traj) ** 2).sum(dim=1)
        #print(loss.shape)
        #quit()
        return loss.mean()

def plot_loss(loss_dict, outer_i):
    """ Log loss to wandb """

    log_dict = {f'{k}': v for k, v in loss_dict.items()}
    log_dict['outer_i'] = outer_i
    wandb.log(log_dict)


def evaluate_action_optimization(conf, learned_cost, irl_loss_fn, mpc, trajs, plots_dir=None, outer_i=None):
    """ Test current learned cost by running inner loop action optimization on test demonstrations """
    # np.random.seed(cfg.random_seed)
    # torch.manual_seed(cfg.random_seed)

    test_pred_trajs = [] # final predicted traj per test traj
    eval_costs = [] # final cost values per test traj

    for t_i, expert_demo_dict in enumerate(trajs):
 
        obs_dict_init = d_utils.get_obs_dict_from_traj(expert_demo_dict, 0, mpc.obj_state_type) # Initial state

        # Reset mpc for action optimization
        mpc.reset_actions()

        action_optimizer = torch.optim.SGD(mpc.parameters(), lr=conf.action_lr)

        for i in range(conf.n_inner_iter):
            action_optimizer.zero_grad()

            pred_traj = mpc.roll_out(obs_dict_init.copy())

            # use the learned loss to update the action sequence
            target = get_target_for_cost_type(expert_demo_dict, conf.cost_type, conf.cost_state, mpc.obj_state_type)
            pred_traj_for_cost = d_utils.parse_pred_traj(pred_traj, conf.cost_state)
            learned_cost_val = learned_cost(pred_traj_for_cost, target)
            learned_cost_val.backward(retain_graph=True)
            action_optimizer.step()

        # Actually take the next step after optimizing the action
        pred_state_traj_new = mpc.roll_out(obs_dict_init.copy())
        eval_costs.append(irl_loss_fn(pred_state_traj_new, expert_demo_dict).mean())

        test_pred_trajs.append(pred_state_traj_new)

        # Plot predicted trajectories
        if plots_dir is not None:
            plot_traj(plots_dir, outer_i, t_i, pred_state_traj_new, expert_demo_dict, conf.mpc_type, mpc.obj_state_type)

    return torch.stack(eval_costs).detach(), test_pred_trajs

def train(conf, learnable_cost, irl_loss_fn, mpc, train_trajs, test_trajs, 
              model_data_dir=None, no_wandb=False):
    """ Training loop for MBIRL """

    irl_loss_on_train = [] # Average IRL loss on training demos, every outer loop
    irl_loss_on_test = []  # Average IRL loss on test demos, every outer loop

    learnable_cost_opt = torch.optim.Adam(learnable_cost.parameters(), lr=conf.cost_lr)

    # Make logging directories
    ckpts_dir = os.path.join(model_data_dir, "ckpts") 
    plots_dir = os.path.join(model_data_dir, "train_trajs")
    test_plots_dir = os.path.join(model_data_dir, "test_trajs")
    if not os.path.exists(ckpts_dir): os.makedirs(ckpts_dir)
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    if not os.path.exists(test_plots_dir): os.makedirs(test_plots_dir)

    #print("Cost function parameters to be optimized:")
    #for name, param in learnable_cost.named_parameters():
    #    print(name)
    #    print(param)

    # Start of inverse RL loop
    for outer_i in range(conf.n_outer_iter):
        irl_loss_per_demo = []
        pred_traj_per_demo = []

        for demo_i in range(len(train_trajs)):
            learnable_cost_opt.zero_grad()
            expert_demo_dict = train_trajs[demo_i]

            obs_dict_init = d_utils.get_obs_dict_from_traj(expert_demo_dict, 0, mpc.obj_state_type) # Initial state

            # Reset mpc for action optimization
            mpc.reset_actions()

            action_optimizer = torch.optim.SGD(mpc.parameters(), lr=conf.action_lr)

            with higher.innerloop_ctx(mpc, action_optimizer) as (fpolicy, diffopt):
                for i in range(conf.n_inner_iter):
                    pred_traj = fpolicy.roll_out(obs_dict_init.copy())
                    # use the learned loss to update the action sequence
                    target = get_target_for_cost_type(expert_demo_dict, conf.cost_type, conf.cost_state, mpc.obj_state_type)
                    pred_traj_for_cost = d_utils.parse_pred_traj(pred_traj, conf.cost_state)
                    learned_cost_val = learnable_cost(pred_traj_for_cost, target)
                    diffopt.step(learned_cost_val)


                # Compute traj with updated action sequence
                pred_traj = fpolicy.roll_out(obs_dict_init.copy())
                # compute task loss
                irl_loss = irl_loss_fn(pred_traj, expert_demo_dict).mean()
                # backprop gradient of learned cost parameters wrt irl loss
                irl_loss.backward(retain_graph=True)

            # Update cost parameters
            learnable_cost_opt.step()

            # Save losses and predicted trajectories
            irl_loss_per_demo.append(irl_loss.detach())
            pred_traj_per_demo.append(pred_traj.detach())

            # Plot
            if (outer_i+1) % conf.n_epoch_every_log == 0:
                plot_traj(plots_dir, outer_i, demo_i, pred_traj, expert_demo_dict, conf.mpc_type, mpc.obj_state_type)
                pred_actions = fpolicy.action_seq.data.detach().numpy()
                plot_actions(plots_dir, outer_i, demo_i, pred_actions, expert_demo_dict)

        irl_loss_on_train.append(torch.Tensor(irl_loss_per_demo).mean())
        print("irl loss (on train) training iter: {} loss: {}".format(outer_i + 1, irl_loss_on_train[-1]))

        # Evaluate current learned cost on test trajectories
        # Plot test traj predictions every 25 steps
        if (outer_i+1) % conf.n_epoch_every_log == 0:
            test_dir_name = test_plots_dir
        else:
            test_dir_name = None
        test_irl_losses, test_pred_trajs = evaluate_action_optimization(conf, learnable_cost.eval(), 
                                                       irl_loss_fn, mpc, test_trajs,
                                                       plots_dir=test_dir_name, outer_i=outer_i)

        irl_loss_on_test.append(test_irl_losses.mean())
        print("irl loss (on test) training iter: {} loss: {}".format(outer_i + 1, irl_loss_on_test[-1]))
        print("")
    
        # Save learned cost weights into dict
        learnable_cost_params = {}
        for name, param in learnable_cost.named_parameters():
            learnable_cost_params[name] = param
        if len(learnable_cost_params) == 0:
            # For RBF Weighted Cost
            for name, param in learnable_cost.weights_fn.named_parameters():
                learnable_cost_params[name] = param

        # Gradient info
        print(f"mpc")
        #print(mpc.phase2_model)
        for name, p in mpc.named_parameters():
        
            #print(name, p.grad.shape)
            if name != "action_seq": continue

            if p.grad is not None:
                grad = p.grad.detach()
                print(name, grad.shape)
                grad_min_mpc = torch.min(torch.abs(grad))
                grad_max_mpc = torch.max(torch.abs(grad))
                grad_norm_mpc = grad.norm()
            else:
                print("No gradient")
                grad_min_mpc = np.nan 
                grad_max_mpc = np.nan 
                grad_norm_mpc = np.nan

        print("cost")
        for p in learnable_cost.parameters():
            if p.grad is not None:
                grad = p.grad.detach()
                print(grad.shape)
                grad_min_cost = torch.min(torch.abs(grad))
                grad_max_cost = torch.max(torch.abs(grad))
                grad_norm_cost = grad.norm()
            else:
                print("No gradient")
                grad_min_cost = np.nan 
                grad_max_cost = np.nan 
                grad_norm_cost = np.nan 

        if not no_wandb:
            # Plot losses with wandb
            loss_dict = {
                        "train_irl_loss": irl_loss_on_train[-1], 
                        "test_irl_loss": irl_loss_on_test[-1], 
                        "grad_min_mpc": grad_min_mpc,
                        "grad_max_mpc": grad_max_mpc,
                        "grad_norm_mpc": grad_norm_mpc,
                        "grad_min_cost": grad_min_cost,
                        "grad_max_cost": grad_max_cost,
                        "grad_norm_cost": grad_norm_cost,
                        }
            plot_loss(loss_dict, outer_i+1)

        # TODO Save checkpoint here
        if (outer_i+1) % conf.n_epoch_every_log == 0:
            torch.save({
                'irl_loss_train_per_demo' : irl_loss_per_demo,
                'irl_loss_test_per_demo'  : test_irl_losses,
                'train_pred_traj_per_demo': pred_traj_per_demo, 
                'test_pred_traj_per_demo' : test_pred_trajs,
                'cost_parameters'         : learnable_cost_params,
                'conf'                    : conf,
            }, f=f'{ckpts_dir}/epoch_{outer_i+1}_ckpt.pth')


def get_target_for_cost_type(demo_dict, cost_type, cost_state, obj_state_type):
    """ Get target from traj_dict for learnable cost function given cost_type  and cost_state """

    expert_demo = get_expert_demo(demo_dict, cost_state, obj_state_type)
    ft_pos_targets_per_mode = torch.Tensor(demo_dict["ft_pos_targets_per_mode"])

    if cost_type in ['Weighted', 'TimeDep', 'RBF']:
        target = expert_demo[-1, :]
    elif cost_type in ['Traj']:
        target = expert_demo
    elif cost_type in ['MPTimeDep']:
        if cost_state == "ftpos":
            target = ft_pos_targets_per_mode
        elif cost_state in ["obj", "ftpos_obj"]:
            # Get init and goal object pose 
            # TODO for now, just get first and last state from demo traj; eventually use goal?? How to do when not using obj pos?
            init = expert_demo[7, :] # Last ind in phase 1 TODO hack
            goal = expert_demo[-1, :]
            target = torch.stack([init, goal])
        else:
            raise ValueError
    else:
        raise ValueError(f'Cost {cost_type} not implemented')

    return target

def get_expert_demo(traj_dict, expert_demo_state, obj_state_type):
    """ Get expert demo from traj_dict, given expert_demo_state """

    if expert_demo_state == "ftpos":
        expert_demo = torch.Tensor(traj_dict["ft_pos_cur"])
    elif expert_demo_state == "ftpos_obj":
        if obj_state_type == "pos":
            o_expert_demo = torch.Tensor(traj_dict["o_pos_cur"])
        elif obj_state_type == "img_r3m":
            o_expert_demo = torch.Tensor(traj_dict["image_60_r3m"])
        else:
            raise ValueError

        expert_demo = torch.cat([torch.Tensor(traj_dict["ft_pos_cur"]),\
                                 torch.Tensor(o_expert_demo)], dim=1)

    elif expert_demo_state == "obj":
        if obj_state_type == "pos":
            expert_demo = torch.Tensor(traj_dict["o_pos_cur"])
        elif obj_state_type == "img_r3m":
            expert_demo = torch.Tensor(traj_dict["image_60_r3m"])
        else:
            raise ValueError
    else:
        raise ValueError(f"{expert_demo_state} is invalid expert_demo_state")
    return expert_demo

def get_mpc(mpc_type, time_horizon):
    """ Get MPC class """

    if mpc_type == "ftpos":
        #return FTPosSim(time_horizon=time_horizon-1)
        return FTPosMPC(time_horizon=time_horizon-1)


    elif mpc_type in ["ftpos_obj_two_phase", "ftpos_obj_learned_only"]:
        # TODO hardcoded
        ## Phase 2 model trained with cropped phase 2 demos
        #phase2_model_path = "/Users/clairelchen/projects/trifinger_claire/trifinger_mbirl/forward_models/runs/phase2_model_nt-100_ost-pos_train-m2/epoch_1500_ckpt.pth"

        ## Phase 2 model trained with full demos
        #phase2_model_path = "trifinger_mbirl/forward_models/runs/phase2_model_nt-100_ost-pos_train-all/epoch_3000_ckpt.pth"

        phase2_model_path = "trifinger_mbirl/forward_models/runs/phase2_model_nt-100_ost-img_r3m/epoch_5000_ckpt.pth"

        if "learned_only" in mpc_type: learned_only = True
        else: learned_only = False
        
        return TwoPhaseMPC(time_horizon-1, phase2_model_path, learned_only=learned_only)
    else:
        raise ValueError(f"{mpc_type} is invalid mpc_type")

def get_learnable_cost(conf, time_horizon, obj_state_type):
    """ Get learnable cost """

    rbf_kernels  = conf.rbf_kernels
    rbf_width    = conf.rbf_width
    cost_type    = conf.cost_type

    ftpos_dim = 9
    
       
    # Set object state dim
    if obj_state_type == "pos":
        o_state_dim = 3 # TODO hardcoded
    elif obj_state_type == "vertices":
        o_state_dim = 8*3 # TODO hardcoded
    elif obj_state_type == "img_r3m":
        o_state_dim = 2048 # TODO hardcoded
    else:
        raise ValueError("Invalid obj_state_type") 

    # Set dimension for cost function, based on cost_state
    if conf.cost_state == "ftpos":
        n_keypt_dim = ftpos_dim
    elif conf.cost_state == "obj":
        n_keypt_dim = o_state_dim
    elif conf.cost_state == "ftpos_obj":
        n_keypt_dim = ftpos_dim + o_state_dim
    else:
        raise ValueError

    # Set learnable cost
    if cost_type == 'Weighted':
        learnable_cost = LearnableWeightedCost(dim=n_keypt_dim)
    elif cost_type == 'TimeDep':
        learnable_cost = LearnableTimeDepWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim)
    elif cost_type == 'RBF':
        learnable_cost = LearnableRBFWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim,
                                                  width=rbf_width, kernels=rbf_kernels)
    elif cost_type == 'Traj':
        learnable_cost = LearnableFullTrajWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim)
    elif cost_type == 'MPTimeDep':
        learnable_cost = LearnableMultiPhaseTimeDepWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim)
    else:
        raise ValueError(f'Cost {cost_type} not implemented')

    return learnable_cost


def plot_traj(plots_dir, outer_i, demo_i, pred_traj, expert_demo_dict, mpc_type, obj_state_type):
    """ Plot predicted and expert trajectories, based on mpc_type """

    traj_dir = os.path.join(plots_dir, f"traj_{demo_i}")
    if not os.path.exists(traj_dir): os.makedirs(traj_dir)

    title = "Predicted trajectories (outer i: {})".format(outer_i)
    save_name = f"epoch_{outer_i+1}.png"
    save_path = os.path.join(traj_dir, save_name)

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

def plot_actions(plots_dir, outer_i, demo_i, pred_actions, expert_demo_dict):
    """ Plot predicted and expert actions """

    expert_actions = expert_demo_dict["delta_ftpos"]
    title = "Fingertip position deltas (outer i: {})".format(outer_i)
    save_name = f"epoch_{outer_i+1}_action.png"
    traj_dir = os.path.join(plots_dir, "actions", f"traj_{demo_i}")
    if not os.path.exists(traj_dir): os.makedirs(traj_dir)
    save_path = os.path.join(traj_dir, save_name)
    d_utils.plot_traj(
            title, 
            save_path,
            ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
            {
            "pred":  {"y": pred_actions, "x": expert_demo_dict["t"][:-1], "marker": "x"},
            #"demo":  {"y": expert_actions[:-1], "x": expert_demo_dict["t"][:-1],"marker": "."},
            }
            )

