#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import sys

sys.path.append("../")
from trifinger_vc.utils import tf_test_util
import numpy as np
from trifinger_vc.algos.bc_finetune import BCFinetune

import random
import torch
from omegaconf import OmegaConf
import json
import os
import trifinger_vc.utils.train_utils as t_utils
from hydra import compose, initialize
from trifinger_vc.utils.sim_nn import Task
import shutil
import hydra
import unittest


class TestTrifingerBC(unittest.TestCase):
    def test_step_reach_env(self):
        env = tf_test_util.init_reach_env()
        env.reset()
        for i in range(9):
            pred_action = np.zeros(3)
            observation, reward, episode_done, info = env.step(pred_action)
            assert reward == 0, "expect 0 reward"

    def test_task(self):
        cfg = tf_test_util.setup_bc_tests()
        cfg["task"]["n_outer_iter"] = 1
        bc = tf_test_util.init_bc_algo(cfg)
        task = tf_test_util.init_reach_task(cfg,bc)
        traj_list = bc.traj_info["test_demos"]
        demo = traj_list[0]
        sim_traj_dict = task.execute_policy(
            bc.policy,
            demo,
            save_dir=tf_test_util.EXP_DIR,
            encoder=bc.encoder,
            epoch=10,
        )

        # assert that the type and shape of the values return match
        assert type(sim_traj_dict) is dict, "Expected dictionary to be returned from execute_policy"

        traj_dict_keys = sim_traj_dict.keys()
        assert "t" in traj_dict_keys, "Expected dictionary to have timestamp"
        assert "o_pos_cur" in traj_dict_keys, "Expect cube pos to be returned"
        assert "robot_pos" in traj_dict_keys, "Expect robot pos to be returned"
        print(sim_traj_dict["robot_pos"])
        print(type(sim_traj_dict["robot_pos"]))

        expected_robot_pos = np.array([[-0.0805528, 1.14, -1.50, -0.08, 1.15, -1.5, -0.08,   1.15,  -1.5 ] \
                               , [-0.07,  1.14, -1.50, -0.07, 1.15, -1.50, -0.08, 1.15, -1.50] \
                                , [-0.07, 1.14, -1.50, -0.07,  1.15, -1.50, -0.08, 1.15, -1.50] \
                                , [-0.06, 1.14, -1.51, -0.07,  1.15, -1.50 , -0.08, 1.150, -1.50] \
                                , [-0.06, 1.14, -1.51, -0.07,  1.15, -1.50 , -0.08, 1.15, -1.50] \
                                , [-0.06, 1.145, -1.51, -0.079, 1.15, -1.50, -0.08, 1.15, -1.50]])

        assert np.all(sim_traj_dict["robot_pos"]- expected_robot_pos < 0.02), "Robot pos not as expected"
        tf_test_util.cleanup_bc_tests()

    def test_bc_algo(self):
        cfg = tf_test_util.setup_bc_tests()
        cfg["task"]["n_outer_iter"] = 1
        bc = tf_test_util.init_bc_algo(cfg)
        bc.train(tf_test_util.EXP_DIR,no_wandb=True)
        
        assert os.path.isdir(os.path.join(tf_test_util.EXP_DIR,"ckpts")), "Expect checkpoints dir to be created."
        assert os.path.isdir(os.path.join(tf_test_util.EXP_DIR,"sim")), "Expect checkpoints dir to be created."
        assert os.path.isfile(os.path.join(tf_test_util.EXP_DIR,"ckpts","epoch_1_ckpt.pth")), "Expected checkpoint file to be saved."
        tf_test_util.cleanup_bc_tests()


    def test_bc_train(self):
        cfg = tf_test_util.setup_bc_tests()

        cfg["task"]["n_outer_iter"] = 10
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Load train and test trajectories
        with open(cfg.task.demo_path, "r") as f:
            traj_info = json.load(f)
        train_traj_stats = traj_info["train_demo_stats"]
        test_traj_stats = traj_info["test_demo_stats"]

        exp_dir = "./test_output"
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # Get traj lists (read from demo files) and add to traj_info
        traj_info["train_demos"] = t_utils.get_traj_list('./assets/',train_traj_stats, "pos")
        traj_info["test_demos"] = t_utils.get_traj_list('./assets/',test_traj_stats, "pos")

        bc = BCFinetune(cfg, traj_info, device)
        bc.policy.train()
        bc.encoder.train()
        for i in range(10):
            train_loss = bc.train_one_epoch()

        assert abs(train_loss - 9.340) < 2, "Training loss not as expected"

        for sim_env_name, sim_params in bc.sim_dict.items():
            sim = Task(
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
            traj_list = bc.traj_info["test_demos"]
            demo = traj_list[0]
            sim_traj_dict = sim.execute_policy(
                bc.policy,
                demo,
                save_dir=exp_dir,
                encoder=bc.encoder,
                epoch=10,
            )
            # one for each time step
            assert (
                abs(sim_traj_dict["scaled_success"].mean() - 0.5) < 0.05
            ), "Mean success of sim rollout not as expected."
            assert (
                abs(sim_traj_dict["scaled_success"][-1] - [0.7222]) < 0.02
            ), "Sim rollout performance not as expected."

            tf_test_util.cleanup_bc_tests()

    def test_bc_train_long(self):
        cfg = tf_test_util.setup_bc_tests()

        cfg["task"]["n_outer_iter"] = 10
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Load train and test trajectories
        with open(cfg.task.demo_path, "r") as f:
            traj_info = json.load(f)
        train_traj_stats = traj_info["train_demo_stats"]
        test_traj_stats = traj_info["test_demo_stats"]

        exp_dir = "./test_output"
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # Get traj lists (read from demo files) and add to traj_info
        traj_info["train_demos"] = t_utils.get_traj_list('./assets/',train_traj_stats, "pos")
        traj_info["test_demos"] = t_utils.get_traj_list('./assets/',test_traj_stats, "pos")

        bc = BCFinetune(cfg, traj_info, device)
        bc.policy.train()
        bc.encoder.train()
        for i in range(50):
            train_loss = bc.train_one_epoch()

        assert abs(train_loss - 2.5) < 1, "Training loss not as expected"

        for sim_env_name, sim_params in bc.sim_dict.items():
            sim = Task(
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
            traj_list = bc.traj_info["test_demos"]
            demo = traj_list[0]
            sim_traj_dict = sim.execute_policy(
                bc.policy,
                demo,
                save_dir=exp_dir,
                encoder=bc.encoder,
                epoch=10,
            )
            assert (
                abs(sim_traj_dict["scaled_success"].mean() - 0.5) < 0.05
            ), "Mean success of sim rollout not as expected."
            assert abs(
                sim_traj_dict["scaled_success"][-1] - [0.8080] < 0.02
            ), "Sim rollout performance not as expected."
            tf_test_util.cleanup_bc_tests()


if __name__ == "__main__":
    unittest.main()
