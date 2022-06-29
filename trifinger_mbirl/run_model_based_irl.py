# Copyright (c) Facebook, Inc. and its affiliates.
import random
import os
import sys
import torch
import numpy as np
import higher
import matplotlib.pyplot as plt
import argparse

from differentiable_robot_model.robot_model import DifferentiableTrifingerEdu
import trifinger_simulation.finger_types_data

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from trifinger_mbirl.ftpos_mpc import FTPosMPC
from trifinger_mbirl.learnable_costs import LearnableWeightedCost, LearnableTimeDepWeightedCost, LearnableRBFWeightedCost
import utils.data_utils as d_utils

# The IRL Loss, the learning objective for the learnable cost functions.
# The IRL loss measures the distance between the demonstrated fingertip position trajectory and predicted trajectory
class IRLLoss(object):
    def __call__(self, pred_traj, target_traj, dist_scale=100):
        loss = ((pred_traj * dist_scale - target_traj * dist_scale) ** 2).sum(dim=0)
        return loss.mean()


def evaluate_action_optimization(learned_cost, robot_model, irl_loss_fn, trajs, n_inner_iter, action_lr=0.001, plots_dir=None):
    # np.random.seed(cfg.random_seed)
    # torch.manual_seed(cfg.random_seed)

    """ Test current learned cost by running inner loop action optimization on test demonstrations """

    eval_costs = []
    for t_i, traj in enumerate(trajs):
 
        x_init   = torch.Tensor(traj["ft_pos_cur"][0, :].squeeze())
        traj_len = traj["ft_pos_cur"].shape[0]
        expert_demo = torch.Tensor(traj["ft_pos_cur"])
        time_horizon, s_dim = expert_demo.shape

        ftpos_mpc = FTPosMPC(time_horizon=time_horizon) # TODO check this (-1??)

        action_optimizer = torch.optim.SGD(ftpos_mpc.parameters(), lr=action_lr)

        for i in range(n_inner_iter):
            action_optimizer.zero_grad()

            pred_traj = ftpos_mpc.roll_out(x_init.clone())
            # use the learned loss to update the action sequence
            learned_cost_val = learned_cost(pred_traj, expert_demo[-1, :])
            learned_cost_val.backward(retain_graph=True)
            action_optimizer.step()

        # Actually take the next step after optimizing the action
        pred_state_traj_new = ftpos_mpc.roll_out(x_init.clone())
        eval_costs.append(irl_loss_fn(pred_state_traj_new, expert_demo).mean())

        title = "Fingertip positions"
        save_name = f"test_{t_i}.png"
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


def irl_training(learnable_cost, robot_model, irl_loss_fn, train_trajs, test_trajs, n_outer_iter, n_inner_iter,
                 cost_type, cost_lr=1e-2, action_lr=1e-3, model_data_dir=None):
    """ Helper function for the irl learning loop """

    irl_loss_on_train = []
    irl_loss_on_test = []

    learnable_cost_opt = torch.optim.Adam(learnable_cost.parameters(), lr=cost_lr)

    irl_loss_per_demo = []

    plots_dir = os.path.join(model_data_dir, f"test_{cost_type}")
    actions_dir = os.path.join(plots_dir, "actions")
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
        ftpos_mpc = FTPosMPC(time_horizon=time_horizon) # TODO check this -1
        pred_traj = ftpos_mpc.roll_out(x_init.clone())

        # get initial irl loss
        irl_loss = irl_loss_fn(pred_traj, expert_demo).mean()
        irl_loss_per_demo.append(irl_loss.item())

    print(irl_loss)

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
            ftpos_mpc = FTPosMPC(time_horizon=time_horizon) # TODO check this (-1??)

            action_optimizer = torch.optim.SGD(ftpos_mpc.parameters(), lr=action_lr)

            with higher.innerloop_ctx(ftpos_mpc, action_optimizer) as (fpolicy, diffopt):
                for i in range(n_inner_iter): # TODO take more gradient descent steps, maybe?
                    pred_traj = fpolicy.roll_out(x_init.clone())

                    # use the learned loss to update the action sequence
                    learned_cost_val = learnable_cost(pred_traj, expert_demo[-1, :])
                    diffopt.step(learned_cost_val)
                    #print(fpolicy.action_seq.data[1, :])
                    actions = fpolicy.action_seq.data.detach().numpy()

                    # Compute traj with updated action sequence
                    pred_traj = fpolicy.roll_out(x_init)
                    # compute task loss
                    irl_loss = irl_loss_fn(pred_traj, expert_demo).mean()
                    # backprop gradient of learned cost parameters wrt irl loss
                    irl_loss.backward(retain_graph=True)
                    irl_loss_per_demo.append(irl_loss.detach())

            learnable_cost_opt.step()


            if outer_i % 25 == 0:
                title = "Fintertip positions (outer i: {})".format(outer_i)
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

                title = "Fingetip position deltas (outer i: {})".format(outer_i)
                save_name = f"{demo_i}_{outer_i}_action.png"
                save_path = os.path.join(actions_dir, save_name)
                d_utils.plot_traj(
                        title, 
                        save_path,
                        ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
                        {
                        "pred":  {"y": actions, "x": expert_demo_dict["t"], "marker": "x"},
                        "demo":  {"y": expert_actions, "x": expert_demo_dict["t"], "marker": "."},
                        }
                        )

        irl_loss_on_train.append(torch.Tensor(irl_loss_per_demo).mean())
        print("irl loss (on train) training iter: {} loss: {}".format(outer_i + 1, irl_loss_on_train[-1]))

        if outer_i % 25 == 0:
            test_irl_losses = evaluate_action_optimization(learnable_cost.eval(), robot_model, irl_loss_fn, test_trajs, n_inner_iter, 
                                                           action_lr=action_lr, plots_dir=plots_dir)

            print("irl loss (on test) training iter: {} loss: {}".format(outer_i + 1, test_irl_losses.mean().item()))

        print("")

        irl_loss_on_test.append(test_irl_losses)
        learnable_cost_params = {}
        for name, param in learnable_cost.named_parameters():
            learnable_cost_params[name] = param

        if len(learnable_cost_params) == 0:
            # For RBF Weighted Cost
            for name, param in learnable_cost.weights_fn.named_parameters():
                learnable_cost_params[name] = param

            
    title = "Fintertip positions (outer i: {})".format(outer_i)
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

    print("Cost function parameters to be optimized:")
    for name, param in learnable_cost.named_parameters():
        print(name)
        print(param)
        # TODO plot costs??

    return torch.stack(irl_loss_on_train), torch.stack(irl_loss_on_test), learnable_cost_params, pred_traj


def main(args):
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)

    robot_model = DifferentiableTrifingerEdu()

    # Load trajectory, get x_init and time_horizon
    data = np.load(args.file_path, allow_pickle=True)["data"]
    traj = d_utils.get_traj_dict_from_obs_list(data)
    
    # Full trajectory, downsampled by 25x (10Hz), cropped out 20 steps
    #traj = d_utils.downsample_traj_dict(traj)
    #traj = d_utils.crop_traj_dict(d_utils.downsample_traj_dict(traj), [10, 30])

    # Full trajectory, downsampled by 75x (3.3Hz)
    traj = d_utils.downsample_traj_dict(traj, new_time_step=0.3)
        
    time_horizon = traj["ft_pos_cur"].shape[0]
    print(time_horizon)
    x_init = traj["ft_pos_cur"][0, :]
    n_keypt_dim = traj["ft_pos_cur"].shape[1] # xyz position of each fingertip

    # type of cost
    #cost_type = 'Weighted'
    #cost_type = 'TimeDep'
    cost_type = 'RBF'

    if cost_type == 'Weighted':
        learnable_cost = LearnableWeightedCost(dim=n_keypt_dim)
    elif cost_type == 'TimeDep':
        learnable_cost = LearnableTimeDepWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim)
    elif cost_type == 'RBF':
        learnable_cost = LearnableRBFWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim)
    else:
        print('Cost not implemented')

    irl_loss_fn = IRLLoss()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    cost_lr      = 1e-2
    action_lr    = 1e-2 # 1e-3
    n_outer_iter = 300
    n_inner_iter = 5 # 1??
    #n_test_traj  = 2
    train_trajs  = [traj]
    test_trajs   = [traj]
    irl_loss_train, irl_loss_test, learnable_cost_params, pred_traj = irl_training(learnable_cost, robot_model,
                                                                                   irl_loss_fn,
                                                                                   train_trajs, test_trajs,
                                                                                   n_outer_iter, n_inner_iter,
                                                                                   cost_type=cost_type,
                                                                                   cost_lr=cost_lr,
                                                                                   action_lr=action_lr, 
                                                                                   model_data_dir=args.log_dir)


    #torch.save({
    #    'irl_loss_train': irl_loss_train,
    #    'irl_loss_test': irl_loss_test,
    #    'cost_parameters': learnable_cost_params,
    #    'final_pred_traj': pred_traj,
    #    'n_inner_iter': n_inner_iter,
    #    'action_lr': action_lr
    #}, f=f'{model_data_dir}/{data_type}_{cost_type}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", default=None, help="""Filepath of trajectory to load""")
    parser.add_argument("--log_dir", type=str, default="/Users/clairelchen/logs/runs/", help="Directory for run logs")
    args = parser.parse_args()
    main(args)

