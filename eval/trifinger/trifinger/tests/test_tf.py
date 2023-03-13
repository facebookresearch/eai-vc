import sys

sys.path.append("../")
from utils import tf_test_util
import numpy as np
from algos.bc_finetune import BCFinetune

import random
import torch
from omegaconf import OmegaConf
import json
import os
import utils.train_utils as t_utils
from hydra import compose, initialize
from utils.sim_nn import SimNN
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

    def test_bc_train(self):
        hydra.core.global_hydra.GlobalHydra.get_state().clear()
        initialize(config_path="../config", job_name="tf_bc")
        cfg = compose(config_name="test_bc")

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
        traj_info["train_demos"] = t_utils.get_traj_list(train_traj_stats, "pos")
        traj_info["test_demos"] = t_utils.get_traj_list(test_traj_stats, "pos")

        bc = BCFinetune(cfg, traj_info, device)
        bc.policy.train()
        bc.encoder.train()
        for i in range(10):
            train_loss = bc.train_one_epoch()

        print(train_loss)
        assert abs(train_loss - 9.340) < 2, "Training loss not as expected"

        for sim_env_name, sim_params in bc.sim_dict.items():
            sim = SimNN(
                bc.policy.in_dim,
                bc.policy.max_a,
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
                demo,
                bc.policy.state_dict(),
                save_dir=exp_dir,
                encoder=bc.encoder,
                epoch=10,
            )
            # one for each time step
            print(sim_traj_dict["scaled_success"])
            assert (
                abs(sim_traj_dict["scaled_success"].mean() - 0.5) < 0.05
            ), "Mean success of sim rollout not as expected."
            assert (
                abs(sim_traj_dict["scaled_success"][-1] - [0.7222]) < 0.02
            ), "Sim rollout performance not as expected."
            shutil.rmtree(exp_dir)

    def test_bc_train_long(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(config_path="../config", job_name="tf_bc")
        cfg = compose(config_name="test_bc")

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
        traj_info["train_demos"] = t_utils.get_traj_list(train_traj_stats, "pos")
        traj_info["test_demos"] = t_utils.get_traj_list(test_traj_stats, "pos")

        bc = BCFinetune(cfg, traj_info, device)
        bc.policy.train()
        bc.encoder.train()
        for i in range(50):
            train_loss = bc.train_one_epoch()
            print(train_loss)

        print(train_loss)
        assert abs(train_loss - 2.5) < 1, "Training loss not as expected"

        for sim_env_name, sim_params in bc.sim_dict.items():
            sim = SimNN(
                bc.policy.in_dim,
                bc.policy.max_a,
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
                demo,
                bc.policy.state_dict(),
                save_dir=exp_dir,
                encoder=bc.encoder,
                epoch=10,
            )
            print(sim_traj_dict["scaled_success"])
            assert (
                abs(sim_traj_dict["scaled_success"].mean() - 0.5) < 0.05
            ), "Mean success of sim rollout not as expected."
            assert abs(
                sim_traj_dict["scaled_success"][-1] - [0.8080] < 0.02
            ), "Sim rollout performance not as expected."
            shutil.rmtree(exp_dir)


if __name__ == "__main__":
    unittest.main()
