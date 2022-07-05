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

import trifinger_simulation.finger_types_data

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from trifinger_mbirl.ftpos_mpc import FTPosMPC
from trifinger_mbirl.learnable_costs import *
import utils.data_utils as d_utils

# The IRL Loss, the learning objective for the learnable cost functions.
# The IRL loss measures the distance between the demonstrated fingertip position trajectory and predicted trajectory
class IRLLoss(object):
    def __call__(self, pred_traj, target_traj, dist_scale=100):
        loss = ((pred_traj * dist_scale - target_traj * dist_scale) ** 2).sum(dim=0)
        return loss.mean()

def plot_loss(loss_dict, outer_i):
    log_dict = {f'{k}': v for k, v in loss_dict.items()}
    log_dict['outer_i'] = outer_i
    wandb.log(log_dict)

def evaluate_action_optimization(conf, learned_cost, irl_loss_fn, trajs, plots_dir=None):
    """ Test current learned cost by running inner loop action optimization on test demonstrations """
    # np.random.seed(cfg.random_seed)
    # torch.manual_seed(cfg.random_seed)

    cost_type      = conf.cost_type
    action_lr      = conf.action_lr
    n_inner_iter   = conf.n_inner_iter
    irl_loss_scale = conf.irl_loss_scale

    eval_costs = []
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

        if plots_dir is not None:
            title = "Fingertip positions"
            save_name = f"test_traj_{t_i}.png"
            save_path = os.path.join(plots_dir, save_name)
            d_utils.plot_traj(
                    title, 
                    save_path,
                    ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
                    {
                    "pred":  {"y": pred_state_traj_new.detach().numpy(), "x": traj["t"], "marker": "x"},
                    "demo":  {"y": expert_demo.detach().numpy(), "x": traj["t"], "marker": "."},
                    }
                    )

    return torch.stack(eval_costs).detach()


def irl_training(conf, learnable_cost, irl_loss_fn, train_trajs, test_trajs, 
                 model_data_dir=None, no_wandb=False):
    """ Helper function for the irl learning loop """

    cost_type      = conf.cost_type
    cost_lr        = conf.cost_lr
    action_lr      = conf.action_lr
    n_outer_iter   = conf.n_outer_iter
    n_inner_iter   = conf.n_inner_iter
    irl_loss_scale = conf.irl_loss_scale

    irl_loss_on_train = []
    irl_loss_on_test = []

    learnable_cost_opt = torch.optim.Adam(learnable_cost.parameters(), lr=cost_lr)

    irl_loss_per_demo = []

    plots_dir = os.path.join(model_data_dir, f"trajs")
    actions_dir = os.path.join(model_data_dir, "actions")
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    if not os.path.exists(actions_dir): os.makedirs(actions_dir)

    # Compute initial loss before training
    for demo_i in range(len(train_trajs)):
        expert_demo_dict = train_trajs[demo_i]

        x_init   = torch.Tensor(expert_demo_dict["ft_pos_cur"][0, :].squeeze())
        traj_len = expert_demo_dict["ft_pos_cur"].shape[0]
        expert_demo = torch.Tensor(expert_demo_dict["ft_pos_cur"])
        time_horizon, s_dim = expert_demo.shape

        # Forward rollout
        ftpos_mpc = FTPosMPC(time_horizon=time_horizon-1)
        pred_traj = ftpos_mpc.roll_out(x_init.clone())

        # get initial irl loss
        irl_loss = irl_loss_fn(pred_traj, expert_demo, dist_scale=irl_loss_scale).mean()
        irl_loss_per_demo.append(irl_loss.item())

    irl_loss_on_train.append(torch.Tensor(irl_loss_per_demo).mean())
    print("irl cost training iter: {} loss: {}".format(0, irl_loss_on_train[-1]))

    print("Cost function parameters to be optimized:")
    for name, param in learnable_cost.named_parameters():
        print(name)
        print(param)

    # Start of inverse RL loop
    for outer_i in range(n_outer_iter):
        irl_loss_per_demo = []

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
                    irl_loss_per_demo.append(irl_loss.detach())

            learnable_cost_opt.step()

            if outer_i % 25 == 0:
                title = "Fingertip positions (outer i: {})".format(outer_i)
                save_name = f"{demo_i}_{outer_i}.png"
                save_path = os.path.join(plots_dir, save_name)
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
                save_name = f"{demo_i}_{outer_i}_action.png"
                save_path = os.path.join(actions_dir, save_name)
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

        # Plot test traj predictions every 25 steps
        if outer_i % 25 == 0:
            test_plots_dir = plots_dir
        else:
            test_plots_dir = None

        test_irl_losses = evaluate_action_optimization(conf, learnable_cost.eval(), 
                                                       irl_loss_fn, test_trajs, plots_dir=test_plots_dir)

        print("irl loss (on test) training iter: {} loss: {}".format(outer_i + 1, test_irl_losses.mean().item()))
        print("")
        irl_loss_on_test.append(test_irl_losses)
    
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

    title = "Fingertip positions (outer i: {})".format(outer_i)
    save_name = f"{demo_i}_{outer_i}.png"
    save_path = os.path.join(plots_dir, save_name)
    d_utils.plot_traj(
            title, 
            save_path,
            ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
            {
            "pred":  {"y": pred_traj.detach().numpy(), "x": expert_demo_dict["t"], "marker": "x"},
            "demo":  {"y": expert_demo.detach().numpy(), "x": expert_demo_dict["t"], "marker": "."},
            }
            )

    return torch.stack(irl_loss_on_train), torch.stack(irl_loss_on_test), learnable_cost_params, pred_traj

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

def main(conf):
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)

    # Load trajectory, get x_init and time_horizon
    data = np.load(conf.file_path, allow_pickle=True)["data"]
    traj = d_utils.get_traj_dict_from_obs_list(data)
    
    # Full trajectory, downsampled by 25x (10Hz), cropped out 20 steps
    #traj = d_utils.downsample_traj_dict(traj)
    #traj = d_utils.crop_traj_dict(d_utils.downsample_traj_dict(traj), [10, 30])

    # Full trajectory, downsampled by 75x (3.3Hz)
    traj = d_utils.downsample_traj_dict(traj, new_time_step=0.3)
        
    time_horizon = traj["ft_pos_cur"].shape[0]
    x_init = traj["ft_pos_cur"][0, :]
    n_keypt_dim = traj["ft_pos_cur"].shape[1] # xyz position of each fingertip

    # type of cost
    #cost_type = 'Weighted'
    #cost_type = 'TimeDep'
    #cost_type = 'RBF'
    #cost_type = 'Traj'
    #cost_type = 'MPTimeDep'

    #exp_params_dict = {
    #                  "cost_type"    : 'Traj',
    #                  "cost_lr"      : 1e-2,
    #                  "action_lr"    : 1e-2, # 1e-3
    #                  "n_outer_iter" : 1500,
    #                  "n_inner_iter" : 10, # 1??
    #                  "rbf_kernels"  : None,
    #                  "rbf_width"    : None,
    #                  "traj_path"    : args.file_path,
    #                  "time_horizon" : time_horizon,
    #                  "irl_loss_scale": 100,
    #                  }

    conf.time_horizon = time_horizon

    rbf_kernels  = conf.rbf_kernels
    rbf_width    = conf.rbf_width
    cost_type    = conf.cost_type

    exp_str = get_exp_str(vars(conf))
    exp_dir = os.path.join(conf.log_dir, exp_str)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if not conf.no_wandb:
        # wandb init
        wandb.init(project='trifinger_mbirl', entity='clairec', name=exp_str, config=conf)

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

    irl_loss_fn = IRLLoss()

    #n_test_traj  = 2
    # For now, just use one traj for training and testing
    train_trajs  = [traj]
    test_trajs   = [traj]
    irl_loss_train, irl_loss_test, learnable_cost_params, pred_traj = irl_training(conf,
                                                                                   learnable_cost, 
                                                                                   irl_loss_fn,
                                                                                   train_trajs, test_trajs,
                                                                                   model_data_dir=exp_dir,
                                                                                   no_wandb=conf.no_wandb,
                                                                                  )

    torch.save({
        'irl_loss_train' : irl_loss_train,
        'irl_loss_test'  : irl_loss_test,
        'cost_parameters': learnable_cost_params,
        'final_pred_traj': pred_traj,
        'conf'           : conf,
    }, f=f'{exp_dir}/log.pth')

def get_exp_str(params_dict):
    
    sorted_dict = collections.OrderedDict(sorted(params_dict.items()))

    run_id = params_dict["run_id"]
    exp_str = f"exp_{run_id}"
    for key, val in sorted_dict.items():
        # TODO shorten traj path to include in exp_str
        if key in ["file_path", "no_wandb", "log_dir", "run_id"]: continue

        # Abbreviate key
        splits = key.split("_")
        short_key = ""
        for split in splits:
            short_key += split[0]
    
        exp_str += "_{}-{}".format(short_key, str(val).replace(".", "p"))

    return exp_str
       

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", default=None, help="""Filepath of trajectory to load""")
    parser.add_argument("--log_dir", type=str, default="/Users/clairelchen/logs/runs/", help="Directory for run logs")
    parser.add_argument("--no_wandb", action="store_true", help="Don't log in wandb")
    parser.add_argument("--run_id", default="NOID", help="Run ID")

    # Parameters
    parser.add_argument("--cost_type", type=str, default="Weighted", help="Learnable cost type")
    parser.add_argument("--cost_lr", type=float, default=0.01, help="Cost learning rate")
    parser.add_argument("--action_lr", type=float, default=0.01, help="Action learning rate")
    parser.add_argument("--n_outer_iter", type=int, default=1500, help="Outer loop iterations")
    parser.add_argument("--n_inner_iter", type=int, default=1, help="Inner loop iterations")
    parser.add_argument("--irl_loss_scale", type=float, default=100, help="IRL loss distance scale")

    # RBF kernel parameters
    parser.add_argument("--rbf_kernels", type=int, default=5, help="Number of RBF kernels")
    parser.add_argument("--rbf_width", type=float, default=2, help="RBF kernel width")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

