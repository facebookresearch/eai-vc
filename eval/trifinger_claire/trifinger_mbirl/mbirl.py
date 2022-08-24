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

from trifinger_mbirl.learnable_costs import *
from trifinger_mbirl.ftpos_mpc import FTPosMPC
from trifinger_mbirl.two_phase_mpc import TwoPhaseMPC
from trifinger_mbirl.learned_mpc import LearnedMPC
from trifinger_mbirl.dynamics_models import FTPosSim
from trifinger_mbirl.forward_models.models.decoder_model import DecoderModel
from trifinger_mbirl.sim_mpc import SimMPC

# A logger for this file
log = logging.getLogger(__name__)

# The IRL Loss, the learning objective for the learnable cost functions.
# Measures the distance between the demonstrated fingertip position trajectory and predicted trajectory
class IRLLoss(object):
    def __init__(self, irl_loss_state, obj_state_type, mpc_use_ftpos):
        self.irl_loss_state = irl_loss_state
        self.obj_state_type = obj_state_type
        self.mpc_use_ftpos = mpc_use_ftpos # True if forward model includes ftpos in state

    def __call__(self, full_pred_traj, expert_demo_dict):
        pred_traj = d_utils.parse_pred_traj(full_pred_traj, self.irl_loss_state, mpc_use_ftpos=self.mpc_use_ftpos)
        target_traj = get_expert_demo(expert_demo_dict, self.irl_loss_state, self.obj_state_type)
        loss = ((pred_traj - target_traj) ** 2).sum(dim=1)
        #print(loss.shape)
        #quit()
        return loss.mean()


class MBIRL:
    def __init__(self, conf, traj_info):

        time_horizon = traj_info["train_demos"][0]["ft_pos_cur"].shape[0]
        
        self.conf = conf
        self.traj_info = traj_info
        self.downsample_time_step = traj_info["downsample_time_step"]

        # Get MPC
        self.mpc = get_mpc(conf.mpc_type, time_horizon, conf.mpc_forward_model_ckpt)
        log.info(f"Loaded MPC type {self.conf.mpc_type} with obj_state_type = {self.mpc.obj_state_type}")

        # Ensure mpc_type and cost/irl_loss states are compatible
        if conf.mpc_type == "learned" and ("ftpos" in conf.cost_state or "ftpos" in conf.irl_loss_state):
            assert mpc.use_ftpos, "Forward model does not use ftpos, but cost_state or irl_loss_state requires ftpos"

        self.mpc_use_ftpos = True
        if conf.mpc_type == "learned" and not self.mpc.use_ftpos: self.mpc_use_ftpos = False
    
        # IRL loss
        self.irl_loss_fn = IRLLoss(conf.irl_loss_state, self.mpc.obj_state_type, self.mpc_use_ftpos)

        # Get learnable cost function
        self.learnable_cost = get_learnable_cost(conf, time_horizon, self.mpc.obj_state_type)

        # Load and use decoder to viz pred_o_states
        if conf.path_to_decoder_ckpt is not None:
            decoder_model_dict = torch.load(conf.path_to_decoder_ckpt) 
            self.decoder = DecoderModel()
            self.decoder.load_state_dict(decoder_model_dict["model_state_dict"])
        else:
            self.decoder = None

        self.sim = SimMPC(downsample_time_step=self.downsample_time_step)

    def train(self, model_data_dir=None, no_wandb=False):
        """ Training loop for MBIRL """

        train_trajs = self.traj_info["train_demos"]
        test_trajs = self.traj_info["test_demos"]

        irl_loss_on_train = [] # Average IRL loss on training demos, every outer loop
        irl_loss_on_test = []  # Average IRL loss on test demos, every outer loop

        learnable_cost_opt = torch.optim.Adam(self.learnable_cost.parameters(), lr=self.conf.cost_lr)

        # Make logging directories
        ckpts_dir = os.path.join(model_data_dir, "ckpts") 
        plots_dir = os.path.join(model_data_dir, "train")
        test_plots_dir = os.path.join(model_data_dir, "test")
        if not os.path.exists(ckpts_dir): os.makedirs(ckpts_dir)
        if not os.path.exists(plots_dir): os.makedirs(plots_dir)
        if not os.path.exists(test_plots_dir): os.makedirs(test_plots_dir)

        #print("Cost function parameters to be optimized:")
        #for name, param in learnable_cost.named_parameters():
        #    print(name)
        #    print(param)

        # Start of inverse RL loop
        for outer_i in range(self.conf.n_outer_iter):
            irl_loss_per_demo = []
            pred_traj_per_demo = []
            pred_actions_per_demo = []

            for demo_i in range(len(train_trajs)):
                learnable_cost_opt.zero_grad()
                expert_demo_dict = train_trajs[demo_i]

                # Init state
                obs_dict_init = d_utils.get_obs_dict_from_traj(expert_demo_dict, 0, self.mpc.obj_state_type) # Initial state

                # Reset mpc for action optimization
                expert_actions = expert_demo_dict["delta_ftpos"][:-1]
                self.mpc.reset_actions()
                #self.mpc.reset_actions(init_a=expert_actions)

                action_optimizer = torch.optim.Adam(self.mpc.parameters(), lr=self.conf.action_lr)
                action_optimizer.zero_grad()
                with higher.innerloop_ctx(self.mpc, action_optimizer) as (fpolicy, diffopt):
                    #################### Inner loop: policy optimization ############################
                    for inner_i in range(self.conf.n_inner_iter):
                        pred_traj = fpolicy.roll_out(obs_dict_init.copy())
                        irl_loss = self.irl_loss_fn(pred_traj, expert_demo_dict).mean()
                        # use the learned loss to update the action sequence
                        target = self.get_target_for_cost_type(expert_demo_dict)
                        pred_traj_for_cost = d_utils.parse_pred_traj(pred_traj, self.conf.cost_state,
                                                                        mpc_use_ftpos=self.mpc_use_ftpos)
                        learned_cost_val = self.learnable_cost(pred_traj_for_cost, target)
                        diffopt.step(learned_cost_val)
                        # Log learned_cost_val, plots epoch_{outer_i}_inner_{inner_i}
                        # TODO for testing; need to make new plot for each traj
                        #print(inner_i)
                        #print(learned_cost_val)
                        #print(fpolicy.action_seq)
                        #if not no_wandb:
                        #    inner_loss_dict = {
                        #        "train_inner_learned_cost_val": learned_cost_val
                        #    }
                        #    t_utils.plot_loss(inner_loss_dict, (outer_i*self.conf.n_inner_iter+inner_i+1))

                        #if (inner_i+1) % 10 == 0:
                        #    diff = self.traj_info["train_demo_stats"][demo_i]["diff"]
                        #    traj_i = self.traj_info["train_demo_stats"][demo_i]["id"]
                        #    traj_plots_dir = os.path.join(plots_dir, f"diff-{diff}_traj-{traj_i}")
                        #    if not os.path.exists(traj_plots_dir): os.makedirs(traj_plots_dir)
                        #    pred_actions = fpolicy.action_seq.data.detach().numpy()
                        #    self.plot(traj_plots_dir, (outer_i*self.conf.n_inner_iter+inner_i), pred_traj, pred_actions, expert_demo_dict)
                        #    
                        #    torch.save({
                        #        'train_pred_traj_per_demo'   : pred_traj.detach(), 
                        #        'train_pred_actions_per_demo': pred_actions,
                        #        'conf'                       : self.conf,
                        #    }, f=f'{ckpts_dir}/epoch_{outer_i*self.conf.n_inner_iter+inner_i+1}_ckpt.pth')
                    #################### End inner loop: policy optimization ############################

                    # Compute traj with updated action sequence
                    pred_traj = fpolicy.roll_out(obs_dict_init.copy())
                    # compute task loss
                    irl_loss = self.irl_loss_fn(pred_traj, expert_demo_dict).mean()
                    # backprop gradient of learned cost parameters wrt irl loss
                    irl_loss.backward(retain_graph=True)

                    pred_actions = fpolicy.action_seq.data.detach().numpy()

                    #self.sim.rollout_actions(expert_demo_dict, pred_actions)

                # Update cost parameters
                learnable_cost_opt.step()

                # Save losses and predicted trajectories
                irl_loss_per_demo.append(irl_loss.detach())
                pred_traj_per_demo.append(pred_traj.detach())
                pred_actions_per_demo.append(pred_actions)

                # Plot
                if (outer_i+1) % self.conf.n_epoch_every_log == 0:
                    diff = self.traj_info["train_demo_stats"][demo_i]["diff"]
                    traj_i = self.traj_info["train_demo_stats"][demo_i]["id"]
                    traj_plots_dir = os.path.join(plots_dir, f"diff-{diff}_traj-{traj_i}")
                    if not os.path.exists(traj_plots_dir): os.makedirs(traj_plots_dir)
                    self.plot(traj_plots_dir, outer_i, pred_traj, pred_actions, expert_demo_dict)

            irl_loss_on_train.append(torch.Tensor(irl_loss_per_demo).mean())
            print("irl loss (on train) training iter: {} loss: {}".format(outer_i + 1, irl_loss_on_train[-1]))

            # Evaluate current learned cost on test trajectories
            # Plot test traj predictions every n_epoch_every_log epochs
            if (outer_i+1) % self.conf.n_epoch_every_log == 0:
                test_dir_name = test_plots_dir
            else:
                test_dir_name = None
            test_irl_losses, test_pred_trajs, test_pred_actions_per_demo = self.evaluate_action_optimization(test_trajs, plots_dir=test_dir_name, outer_i=outer_i)

            irl_loss_on_test.append(test_irl_losses.mean())
            print("irl loss (on test) training iter: {} loss: {}".format(outer_i + 1, irl_loss_on_test[-1]))
            print("")

            # Save learned cost weights into dict
            learnable_cost_params = {}
            for name, param in self.learnable_cost.named_parameters():
                learnable_cost_params[name] = param
            if len(learnable_cost_params) == 0:
                # For RBF Weighted Cost
                for name, param in self.learnable_cost.weights_fn.named_parameters():
                    learnable_cost_params[name] = param

            # Gradient info
            grad_dict = self.get_grad_dict()

            if not no_wandb:
                # Plot losses with wandb
                loss_dict = {
                            "train_irl_loss": irl_loss_on_train[-1], 
                            "test_irl_loss": irl_loss_on_test[-1], 
                            }
                all_dict = {**loss_dict, **grad_dict}
                t_utils.plot_loss(all_dict, outer_i+1)

            if (outer_i+1) % self.conf.n_epoch_every_log == 0:
                torch.save({
                    'irl_loss_train_per_demo'    : irl_loss_per_demo,
                    'irl_loss_test_per_demo'     : test_irl_losses,
                    'train_pred_traj_per_demo'   : pred_traj_per_demo, 
                    'test_pred_traj_per_demo'    : test_pred_trajs,
                    'train_pred_actions_per_demo': pred_actions_per_demo,
                    'test_pred_actions_per_demo' : test_pred_actions_per_demo,
                    'cost_parameters'            : learnable_cost_params,
                    'conf'                       : self.conf,
                }, f=f'{ckpts_dir}/epoch_{outer_i+1}_ckpt.pth')


    def evaluate_action_optimization(self, trajs, plots_dir=None, outer_i=None):
        """ Test current learned cost by running inner loop action optimization on test demonstrations """
        # np.random.seed(cfg.random_seed)
        # torch.manual_seed(cfg.random_seed)

        test_pred_trajs = [] # final predicted traj per test traj
        test_pred_actions = [] # final predicted actions per test traj
        eval_costs = [] # final cost values per test traj

        for t_i, expert_demo_dict in enumerate(trajs):
            obs_dict_init = d_utils.get_obs_dict_from_traj(expert_demo_dict, 0, self.mpc.obj_state_type) # Initial state

            # Reset mpc for action optimization
            expert_actions = expert_demo_dict["delta_ftpos"][:-1]
            self.mpc.reset_actions()
            #self.mpc.reset_actions(init_a=expert_actions)

            #action_optimizer = torch.optim.SGD(self.mpc.parameters(), lr=self.conf.action_lr)
            action_optimizer = torch.optim.Adam(self.mpc.parameters(), lr=self.conf.action_lr)

            for inner_i in range(self.conf.n_inner_iter):
                action_optimizer.zero_grad()

                pred_traj = self.mpc.roll_out(obs_dict_init.copy())

                # use the learned loss to update the action sequence
                target = self.get_target_for_cost_type(expert_demo_dict)
                pred_traj_for_cost = d_utils.parse_pred_traj(pred_traj, self.conf.cost_state,
                                                                mpc_use_ftpos=self.mpc_use_ftpos)
                learned_cost_val = self.learnable_cost(pred_traj_for_cost, target) # TODO check that correct cost is being used
                learned_cost_val.backward(retain_graph=True)
                action_optimizer.step()

            # Actually take the next step after optimizing the action
            pred_state_traj_new = self.mpc.roll_out(obs_dict_init.copy())
            eval_costs.append(self.irl_loss_fn(pred_state_traj_new, expert_demo_dict).mean())

            pred_actions = self.mpc.action_seq.data.detach().numpy()

            test_pred_trajs.append(pred_state_traj_new)
            test_pred_actions.append(pred_actions)

            # Plot predicted trajectories
            if plots_dir is not None:
                diff = self.traj_info["test_demo_stats"][t_i]["diff"]
                traj_i = self.traj_info["test_demo_stats"][t_i]["id"]
                traj_plots_dir = os.path.join(plots_dir, f"diff-{diff}_traj-{traj_i}")
                if not os.path.exists(traj_plots_dir): os.makedirs(traj_plots_dir)
                self.plot(traj_plots_dir, outer_i, pred_state_traj_new, pred_actions, expert_demo_dict)
        
        return torch.stack(eval_costs).detach(), test_pred_trajs, test_pred_actions


    def get_target_for_cost_type(self, demo_dict):
        """ Get target from traj_dict for learnable cost function given cost_type  and cost_state """

        cost_type = self.conf.cost_type
        cost_state = self.conf.cost_state
        obj_state_type = self.mpc.obj_state_type

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

    def plot(self, traj_plots_dir, outer_i, pred_traj, pred_actions, expert_demo_dict):

        if self.mpc_use_ftpos:
            plot_traj(traj_plots_dir, outer_i, pred_traj, expert_demo_dict, self.conf.mpc_type, self.mpc.obj_state_type)

        if self.conf.mpc_type=="learned" and self.decoder is not None:
            ft_states, o_states = self.mpc.get_states_from_x_next(pred_traj)
            with torch.no_grad(): # TODO need to freeze decoder weights??
                pred_imgs = self.decoder(torch.Tensor(o_states))
                self.decoder.save_gif(pred_imgs, os.path.join(traj_plots_dir, f'r3m_epoch_{outer_i+1}.gif'))

        plot_actions(traj_plots_dir, outer_i, pred_actions, expert_demo_dict)

    def get_grad_dict(self):
        return {} # TODO don't log gradients, no support for nn policy

        # Gradient info
        for name, p in self.mpc.named_parameters():
        
            #print(name, p.grad.shape)
            if name != "action_seq": continue

            if p.grad is not None:
                grad = p.grad.detach()
                #print(name, grad.shape)
                grad_min_mpc = torch.min(torch.abs(grad))
                grad_max_mpc = torch.max(torch.abs(grad))
                grad_norm_mpc = grad.norm()
            else:
                #print("No gradient")
                grad_min_mpc = np.nan 
                grad_max_mpc = np.nan 
                grad_norm_mpc = np.nan

        #print("cost")
        for p in self.learnable_cost.parameters():
            if p.grad is not None:
                grad = p.grad.detach()
                #print(grad.shape)
                grad_min_cost = torch.min(torch.abs(grad))
                grad_max_cost = torch.max(torch.abs(grad))
                grad_norm_cost = grad.norm()
            else:
                #print("No gradient")
                grad_min_cost = np.nan 
                grad_max_cost = np.nan 
                grad_norm_cost = np.nan 

        grad_dict = {
                    "grad_min_mpc": grad_min_mpc,
                    "grad_max_mpc": grad_max_mpc,
                    "grad_norm_mpc": grad_norm_mpc,
                    "grad_min_cost": grad_min_cost,
                    "grad_max_cost": grad_max_cost,
                    "grad_norm_cost": grad_norm_cost,
                    }
        return grad_dict

def get_mpc(mpc_type, time_horizon, mpc_forward_model_ckpt=None):
    """ Get MPC class """

    if mpc_type == "ftpos":
        #return FTPosSim(time_horizon=time_horizon-1)
        return FTPosMPC(time_horizon=time_horizon-1)

    elif mpc_type == "two_phase":
        if mpc_forward_model_ckpt is None: raise ValueError("Missing mpc_forward_model_ckpt")

        ## Phase 2 model trained with cropped phase 2 demos
        #phase2_model_path = "/Users/clairelchen/projects/trifinger_claire/trifinger_mbirl/forward_models/runs/phase2_model_nt-100_ost-pos_train-m2/epoch_1500_ckpt.pth"
        ## Phase 2 model trained with full demos
        #phase2_model_path = "trifinger_mbirl/forward_models/runs/phase2_model_nt-100_ost-pos_train-all/epoch_3000_ckpt.pth"

        return TwoPhaseMPC(time_horizon-1, mpc_forward_model_ckpt)

    elif mpc_type == "learned":
        model_dict = torch.load(mpc_forward_model_ckpt) 
        return LearnedMPC(time_horizon-1, model_dict=model_dict)

    else:
        raise ValueError(f"{mpc_type} is invalid mpc_type")

def get_learnable_cost(conf, time_horizon, obj_state_type):
    """ Get learnable cost """

    rbf_kernels  = conf.rbf_kernels
    rbf_width    = conf.rbf_width
    cost_type    = conf.cost_type

    ftpos_dim = 9

    # Set object state dim
    if conf.cost_state in ["ftpos_obj", "obj"]:
        if obj_state_type == "pos":
            o_state_dim = 3 # TODO hardcoded
        elif obj_state_type == "vertices":
            o_state_dim = 8*3 # TODO hardcoded
        elif obj_state_type == "img_r3m":
            o_state_dim = 2048 # TODO hardcoded
        else:
            raise ValueError("Invalid obj_state_type") 
    else:
        o_state_dim = 0

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
