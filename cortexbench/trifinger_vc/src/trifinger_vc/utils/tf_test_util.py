#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import hydra
from hydra import compose, initialize
import random
import numpy as np
import json
import torch
import os
from trifinger_simulation.trifinger_platform import ObjectType
import trifinger_vc.utils.train_utils as t_utils
from trifinger_vc.algos.bc_finetune import BCFinetune
from trifinger_vc.utils.sim_nn import Task
from trifinger_vc.trifinger_envs.action import ActionType
from trifinger_vc.trifinger_envs.cube_reach import CubeReachEnv

EXP_DIR = "./test_output"

def init_reach_env():
    sim_time_step = 0.004
    downsample_time_step = 0.2
    traj_scale = 1
    n_fingers_to_move = 1
    a_dim = n_fingers_to_move * 3
    task = "reach_cube"
    state_type = "ftpos_obj"
    # obj_state_type = "mae_vit_base_patch16_ego4d_210_epochs"
    goal_type = "goal_none"

    step_size = int(downsample_time_step / sim_time_step)
    object_type = ObjectType.COLORED_CUBE
    env = CubeReachEnv(
                action_type=ActionType.TORQUE,
                step_size=step_size,
                visualization=False,
                enable_cameras=True,
                finger_type="trifingerpro",
                camera_delay_steps=0,
                time_step=sim_time_step,
                object_type=object_type,
                enable_shadows=False,
                camera_view="default",
                arena_color="default",
                visual_observation=True,
                run_rl_policy=False,
            )
    return env



def init_bc_algo(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load train and test trajectories
    with open(cfg.task.demo_path, "r") as f:
        traj_info = json.load(f)
    train_traj_stats = traj_info["train_demo_stats"]
    test_traj_stats = traj_info["test_demo_stats"]
    
    # Get traj lists (read from demo files) and add to traj_info
    traj_info["train_demos"] = t_utils.get_traj_list('./assets/',train_traj_stats, "pos")
    traj_info["test_demos"] = t_utils.get_traj_list('./assets/',test_traj_stats, "pos")
    bc = BCFinetune(cfg, traj_info, device)
    bc.policy.train()
    bc.encoder.train()
    return bc


def init_reach_task(cfg, bc):
    sim_params = list(bc.sim_dict.items())[0][1]
    task = Task(
        bc.conf.task.state_type,
        bc.algo_conf.pretrained_rep,  # obj_state_type
        downsample_time_step=bc.traj_info["downsample_time_step"],
        traj_scale=bc.traj_info["scale"],
        goal_type=bc.conf.task.goal_type,
        object_type=bc.traj_info["object_type"],
        finger_type=bc.traj_info["finger_type"],
        enable_shadows=sim_params["enable_shadows"],
        camera_view=sim_params["camera_view"],
        arena_color=sim_params["arena_color"],
        task=bc.task,
        n_fingers_to_move=bc.n_fingers_to_move,
    )
    return task


def setup_bc_tests():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../config", job_name="tf_bc")
    cfg = compose(config_name="test_bc")

    cfg["task"]["n_outer_iter"] = 10

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)
    return cfg

def cleanup_bc_tests():
    shutil.rmtree(EXP_DIR)
