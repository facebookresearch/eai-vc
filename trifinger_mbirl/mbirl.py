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

#import trifinger_simulation.finger_types_data

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from trifinger_mbirl.ftpos_mpc import FTPosMPC
from trifinger_mbirl.learnable_costs import *
import utils.data_utils as d_utils

# The IRL Loss, the learning objective for the learnable cost functions.
# Measures the distance between the demonstrated fingertip position trajectory and predicted trajectory
class IRLLoss(object):
    def __call__(self, pred_traj, target_traj, dist_scale=100):
        loss = ((pred_traj * dist_scale - target_traj * dist_scale) ** 2).sum(dim=0)
        return loss.mean()

def plot_loss(loss_dict, outer_i):
    log_dict = {f'{k}': v for k, v in loss_dict.items()}
    log_dict['outer_i'] = outer_i
    wandb.log(log_dict)

def evaluate_action_optimization(conf, learned_cost, irl_loss_fn, trajs, plots_dir=None, outer_i=None):
    """ Test current learned cost by running inner loop action optimization on test demonstrations """
    # np.random.seed(cfg.random_seed)
    # torch.manual_seed(cfg.random_seed)

    cost_type      = conf.cost_type
    action_lr      = conf.action_lr
    n_inner_iter   = conf.n_inner_iter
    irl_loss_scale = conf.irl_loss_scale

    test_pred_trajs = [] # final predicted traj per test traj
    eval_costs = [] # final cost values per test traj

    for t_i, traj in enumerate(trajs):
 
        x_init   = torch.Tensor(traj["ft_pos_cur"][0, :].squeeze())
        traj_len = traj["ft_pos_cur"].shape[0]
        expert_demo = torch.Tensor(traj["ft_pos_cur"])
        time_horizon, s_dim = expert_demo.shape

        ftpos_mpc = FTPosMPC(time_horizon=time_horizon-1)

        action_optimizer = torch.optim.SGD(ftpos_mpc.parameters(), lr=action_lr)

        for i in range(n_inner_iter):
            action_optimizer.zero_grad()

            pred_traj = ftpos_mpc.roll_out(x_init.clone())

            # use the learned loss to update the action sequence
            target = get_target_for_cost_type(traj, cost_type)
            learned_cost_val = learned_cost(pred_traj, target)
            learned_cost_val.backward(retain_graph=True)
            action_optimizer.step()

        # Actually take the next step after optimizing the action
        pred_state_traj_new = ftpos_mpc.roll_out(x_init.clone())
        eval_costs.append(irl_loss_fn(pred_state_traj_new, expert_demo, dist_scale=irl_loss_scale).mean())

        test_pred_trajs.append(pred_state_traj_new)

        if plots_dir is not None:
            title = "Fingertip positions"
            traj_dir = os.path.join(plots_dir, f"traj_{t_i}")
            if not os.path.exists(traj_dir): os.makedirs(traj_dir)
            save_name = f"epoch_{outer_i}.png"
            save_path = os.path.join(traj_dir, save_name)
            d_utils.plot_traj(
                    title, 
                    save_path,
                    ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
                    {
                    "pred":  {"y": pred_state_traj_new.detach().numpy(), "x": traj["t"], "marker": "x"},
                    "demo":  {"y": expert_demo.detach().numpy(), "x": traj["t"], "marker": "."},
                    }
                    )

    return torch.stack(eval_costs).detach(), test_pred_trajs


def train(conf, learnable_cost, irl_loss_fn, train_trajs, test_trajs, 
              model_data_dir=None, no_wandb=False):
    """ Helper function for the irl learning loop """

    cost_type      = conf.cost_type
    cost_lr        = conf.cost_lr
    action_lr      = conf.action_lr
    n_outer_iter   = conf.n_outer_iter
    n_inner_iter   = conf.n_inner_iter
    irl_loss_scale = conf.irl_loss_scale

    irl_loss_on_train = [] # Average IRL loss on training demos, every outer loop
    irl_loss_on_test = []  # Average IRL loss on test demos, every outer loop

    learnable_cost_opt = torch.optim.Adam(learnable_cost.parameters(), lr=cost_lr)

    # Make logging directories
    ckpts_dir = os.path.join(model_data_dir, "ckpts") 
    plots_dir = os.path.join(model_data_dir, "train_trajs")
    test_plots_dir = os.path.join(model_data_dir, "test_trajs")
    if not os.path.exists(ckpts_dir): os.makedirs(ckpts_dir)
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    if not os.path.exists(test_plots_dir): os.makedirs(test_plots_dir)

    print("Cost function parameters to be optimized:")
    for name, param in learnable_cost.named_parameters():
        print(name)
        print(param)

    # Start of inverse RL loop
    for outer_i in range(n_outer_iter):
        irl_loss_per_demo = []
        pred_traj_per_demo = []

        for demo_i in range(len(train_trajs)):
            learnable_cost_opt.zero_grad()
            expert_demo_dict = train_trajs[demo_i]

            x_init   = torch.Tensor(expert_demo_dict["ft_pos_cur"][0, :].squeeze())
            traj_len = expert_demo_dict["ft_pos_cur"].shape[0]
            expert_demo = torch.Tensor(expert_demo_dict["ft_pos_cur"])
            expert_actions = expert_demo_dict["delta_ftpos"]
            time_horizon, s_dim = expert_demo.shape

            # Forward rollout
            ftpos_mpc = FTPosMPC(time_horizon=time_horizon-1)

            action_optimizer = torch.optim.SGD(ftpos_mpc.parameters(), lr=action_lr)

            with higher.innerloop_ctx(ftpos_mpc, action_optimizer) as (fpolicy, diffopt):
                for i in range(n_inner_iter):
                    pred_traj = fpolicy.roll_out(x_init.clone())

                    # use the learned loss to update the action sequence
                    target = get_target_for_cost_type(expert_demo_dict, cost_type)
                    learned_cost_val = learnable_cost(pred_traj, target)

                    diffopt.step(learned_cost_val)
                    #print(fpolicy.action_seq.data[1, :])

                actions = fpolicy.action_seq.data.detach().numpy()
                # Compute traj with updated action sequence
                pred_traj = fpolicy.roll_out(x_init)
                # compute task loss
                irl_loss = irl_loss_fn(pred_traj, expert_demo, dist_scale=irl_loss_scale).mean()
                # backprop gradient of learned cost parameters wrt irl loss
                irl_loss.backward(retain_graph=True)

            # Update cost parameters
            learnable_cost_opt.step()

            # Save losses and predicted trajectories
            irl_loss_per_demo.append(irl_loss.detach())
            pred_traj_per_demo.append(pred_traj.detach())

            if (outer_i+1) % conf.n_epoch_every_log == 0:
                traj_dir = os.path.join(plots_dir, f"traj_{demo_i}")
                if not os.path.exists(traj_dir): os.makedirs(traj_dir)

                title = "Fingertip positions (outer i: {})".format(outer_i)
                save_name = f"epoch_{outer_i+1}.png"
                save_path = os.path.join(traj_dir, save_name)
                d_utils.plot_traj(
                        title, 
                        save_path,
                        ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
                        {
                        "pred":  {"y": pred_traj.detach().numpy(), "x": expert_demo_dict["t"], "marker": "x"},
                        "demo":  {"y": expert_demo.detach().numpy(), "x": expert_demo_dict["t"], "marker": "."},
                        }
                        )

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
                        "pred":  {"y": actions, "x": expert_demo_dict["t"][:-1], "marker": "x"},
                        "demo":  {"y": expert_actions[:-1], "x": expert_demo_dict["t"][:-1],"marker": "."},
                        }
                        )


        irl_loss_on_train.append(torch.Tensor(irl_loss_per_demo).mean())
        print("irl loss (on train) training iter: {} loss: {}".format(outer_i + 1, irl_loss_on_train[-1]))

        # Evaluate current learned cost on test trajectories
        # Plot test traj predictions every 25 steps
        if (outer_i+1) % conf.n_epoch_every_log == 0:
            test_dir_name = test_plots_dir
        else:
            test_dir_name = None
        test_irl_losses, test_pred_trajs = evaluate_action_optimization(conf, learnable_cost.eval(), 
                                                       irl_loss_fn, test_trajs,
                                                       plots_dir=test_dir_name, outer_i=outer_i+1)

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

        if not conf.no_wandb:
            # Plot losses with wandb
            loss_dict = {
                        "train_irl_loss": irl_loss_on_train[-1], 
                        "test_irl_loss": irl_loss_on_test[-1], 
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


def get_target_for_cost_type(demo_dict, cost_type):
    """ Get target from traj_dict for learnable cost function given cost_type """

    expert_demo = torch.Tensor(demo_dict["ft_pos_cur"])
    ft_pos_targets_per_mode = torch.Tensor(demo_dict["ft_pos_targets_per_mode"])

    if cost_type in ['Weighted', 'TimeDep', 'RBF']:
        target = expert_demo[-1, :]
    elif cost_type in ['Traj']:
        target = expert_demo
    elif cost_type in ['MPTimeDep']:
        target = ft_pos_targets_per_mode
    else:
        raise ValueError(f'Cost {cost_type} not implemented')

    return target

