# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
import torch
import numpy as np
import logging

import trifinger_vc.utils.train_utils as t_utils
import trifinger_vc.utils.data_utils as d_utils
import trifinger_vc.control.cube_utils as c_utils

from trifinger_vc.utils.policy import construct_policy
from trifinger_vc.utils.dataset import BCFinetuneDataset
from trifinger_vc.utils.encoder_model import EncoderModel
from trifinger_vc.utils.sim_nn import Task

# A logger for this file
log = logging.getLogger(__name__)

class BCFinetune:
    def __init__(self, conf, traj_info, device):
        self.conf = conf
        self.algo_conf = conf.algo
        self.traj_info = traj_info
        self.device = device

        # Get task name
        self.task = self.conf.task.name
        
        if(self.task == "reach_cube"):
            fingers_to_move = 1
        elif (self.task == "move_cube"):
            fingers_to_move = 3
        else:
            fingers_to_move = 0

        # Make dataset and dataloader
        train_dataset = BCFinetuneDataset(
            self.traj_info["train_demo_stats"],
            state_type=self.conf.task.state_type,
            obj_state_type=self.algo_conf.pretrained_rep,
            device=self.device,
            augment_prob=self.algo_conf.image_aug_dict["augment_prob"],
            times_to_use_demo=self.algo_conf.image_aug_dict["times_to_use_demo"],
            jitter_brightness=self.algo_conf.image_aug_dict["jitter_brightness"],
            jitter_contrast=self.algo_conf.image_aug_dict["jitter_contrast"],
            jitter_saturation=self.algo_conf.image_aug_dict["jitter_saturation"],
            jitter_hue=self.algo_conf.image_aug_dict["jitter_hue"],
            shift_pad=self.algo_conf.image_aug_dict["shift_pad"],
            fingers_to_move=fingers_to_move
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.algo_conf.batch_size, shuffle=True
        )

        log.info(
            f"Loaded training dataset with {train_dataset.n_augmented_samples} / {len(train_dataset)} augmented samples"
        )

        test_dataset = BCFinetuneDataset(
            self.traj_info["test_demo_stats"],
            state_type=self.conf.task.state_type,
            obj_state_type=self.algo_conf.pretrained_rep,
            device=self.device,
            augment_prob=self.algo_conf.image_aug_dict["augment_prob"],
            times_to_use_demo=1,
            jitter_brightness=self.algo_conf.image_aug_dict["jitter_brightness"],
            jitter_contrast=self.algo_conf.image_aug_dict["jitter_contrast"],
            jitter_saturation=self.algo_conf.image_aug_dict["jitter_saturation"],
            jitter_hue=self.algo_conf.image_aug_dict["jitter_hue"],
            shift_pad=self.algo_conf.image_aug_dict["shift_pad"],
            fingers_to_move=fingers_to_move
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.algo_conf.batch_size, shuffle=True
        )

        self.n_fingers_to_move = train_dataset.n_fingers_to_move

        log.info(
            f"Loaded test dataset with {test_dataset.n_augmented_samples} / {len(test_dataset)} augmented samples"
        )

        # Encoder
        self.encoder = EncoderModel(
            pretrained_rep=self.algo_conf.pretrained_rep,
            freeze_pretrained_rep=self.algo_conf.freeze_pretrained_rep,
            rep_to_policy=self.conf.rep_to_policy,
        ).to(self.device)
        log.info(f"Model:\n{self.encoder}")

        if "obj" in self.conf.task.state_type:
            self.state_type_key = "o_state"
        elif self.conf.task.state_type ==  "goal_cond":
            self.state_type_key = "o_goal"
        # Policy
        self.policy = construct_policy(self.conf.task.state_type, train_dataset[0]["input"]["ft_state"].shape[0], self.encoder.pretrained_rep_dim,
                                              self.conf.task.goal_type,train_dataset.out_dim,self.traj_info["max_a"],self.device)
        self.policy.eval()
        log.info(f"Policy:\n{self.policy}")

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.encoder.parameters(), "lr": self.algo_conf.visual_lr},
                {"params": self.policy.parameters(), "lr": self.algo_conf.lr},
            ],
            lr=self.algo_conf.lr,
            weight_decay=self.algo_conf.adam_weight_decay,
        )

        self.loss_fn = torch.nn.MSELoss()

        # Load sim env for rollouts
        self.run_sim = self.algo_conf.run_sim

        all_sim_dict = {
            "sim_env_demo": {
                "enable_shadows": False,
                "camera_view": "default",
                "arena_color": "default",
            },
            "sim_env_real": {
                "enable_shadows": True,
                "camera_view": "real",
                "arena_color": "real",
            },
            # "sim_env_shadows": {
            #    "enable_shadows": True,
            #    "camera_view": "default",
            #    "arena_color": "default",
            # },
            "sim_env_real_camera_view": {
                "enable_shadows": False,
                "camera_view": "real",
                "arena_color": "default",
            },
            "sim_env_real_arena_color": {
                "enable_shadows": False,
                "camera_view": "default",
                "arena_color": "real",
            },
        }

        self.sim_dict = {}
        for env_name in conf.eval_envs:
            self.sim_dict[env_name] = all_sim_dict[env_name]

    def train_one_epoch(self):
        total_train_loss = 0.0
        for i_batch, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            # First, pass img and goal img through encoder
            if "obj" in self.conf.task.state_type:
                latent_state = self.encoder(batch["input"]["rgb_img_preproc"])
                batch["input"]["o_state"] = latent_state

            if self.conf.task.goal_type == "goal_cond":
                latent_goal = self.encoder(batch["input"]["rgb_img_preproc_goal"])
                batch["input"]["o_goal"] = latent_goal
            # Then, make observation pass through policy
            obs_vec = t_utils.get_bc_obs_vec_from_obs_dict(
                batch["input"], self.conf.task.state_type, self.conf.task.goal_type
            )
            pred_actions = self.policy(obs_vec)

            loss = self.loss_fn(pred_actions, batch["output"]["action"])
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()
        return total_train_loss

    def get_latent_rep(self, batch_input):
        # if "obj" in self.conf.task.state_type
        if self.state_type_key == "o_state":
            latent = self.encoder(batch_input["rgb_img_preproc"])
        if self.state_type_key == "o_goal":
            latent = self.encoder(batch_input["rgb_img_preproc_goal"])
        return latent

    def train(self, model_data_dir=None, no_wandb=False):
        # Make logging directories
        ckpts_dir = os.path.join(model_data_dir, "ckpts")
        sim_dir = os.path.join(model_data_dir, "sim")
        if not os.path.exists(ckpts_dir):
            os.makedirs(ckpts_dir)
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)

        # Search for most recent ckpt in ckpts_dir
        ckpt_pth_to_load, start_epoch = t_utils.find_most_recent_ckpt(ckpts_dir)
        if ckpt_pth_to_load is not None:
            ckpt_info = torch.load(ckpt_pth_to_load)
            self.optimizer.load_state_dict(ckpt_info["optimizer_state_dict"])
            self.policy.load_state_dict(ckpt_info["policy"])
            self.encoder.load_state_dict(ckpt_info["encoder"])
            log.info(f"Loading state from {ckpt_pth_to_load}.")

        # initializing dictionary that keeps track of max values
        # Todo - if we load from checkpoint this max dict needs to be loaded as well
        self.max_dict = {}
        for sim_env_name in self.sim_dict.keys():
            self.max_dict[sim_env_name] = {"train": {}, "test": {}}

        # Sim rollout
        if self.run_sim:
            sim_log_dict = self.sim_rollout(
                sim_dir,
                start_epoch,
                max_demo_per_diff=self.algo_conf.max_demo_per_diff,
            )
            log.info(sim_log_dict)

            if not no_wandb:
                all_dict = {**sim_log_dict}
                t_utils.plot_loss(all_dict, start_epoch + 1)
        for outer_i in range(start_epoch, self.conf.task.n_outer_iter):
            # Update policy network
            self.policy.train()
            self.encoder.train()
            total_train_loss = self.train_one_epoch()

            avg_train_loss = total_train_loss / len(self.train_dataloader)

            # Test
            self.policy.eval()
            self.encoder.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for i_batch, batch in enumerate(self.test_dataloader):
                    self.optimizer.zero_grad()

                    # First, pass img and goal img through encoder
                    batch["input"][self.state_type_key] = self.get_latent_rep(batch["input"])
                    if self.conf.task.goal_type == "goal_cond":
                        latent_goal = self.encoder(batch["input"]["rgb_img_preproc_goal"])
                        batch["input"]["o_goal"] = latent_goal

                    # Then, make observation pass through policy
                    obs_vec = t_utils.get_bc_obs_vec_from_obs_dict(
                        batch["input"],
                        self.conf.task.state_type,
                        self.conf.task.goal_type,
                    )
                    pred_actions = self.policy(obs_vec)

                    loss = self.loss_fn(pred_actions, batch["output"]["action"])

                    total_test_loss += loss.item()

            avg_test_loss = total_test_loss / (i_batch + 1)

            loss_dict = {
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "epoch": outer_i,
            }
            sim_log_dict = {}

            log.info(f"Epoch: {outer_i}, loss: {avg_train_loss}")

            if (outer_i + 1) % self.conf.task.n_epoch_every_log == 0:
                # Sim rollout
                if self.run_sim:
                    sim_log_dict = self.sim_rollout(
                        sim_dir,
                        outer_i,
                        max_demo_per_diff=self.algo_conf.max_demo_per_diff,
                    )
                    log.info(sim_log_dict)

            if not no_wandb:
                all_dict = {**loss_dict, **sim_log_dict}
                t_utils.plot_loss(all_dict, outer_i + 1)

        torch.save(
            {
                "loss_train": avg_train_loss,
                "loss_test": avg_test_loss,
                "policy": self.policy.state_dict(),
                "encoder": self.encoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "conf": self.algo_conf,
                "in_dim": self.policy.in_dim,
                "out_dim": self.policy.out_dim,
            },
            f=f"{ckpts_dir}/epoch_{outer_i+1}_ckpt.pth",
        )

    def sim_rollout(self, sim_dir, outer_i, max_demo_per_diff=10):
        """Rollout policies for train and test demos"""

        log_dict = {}
        for sim_env_name in self.sim_dict.keys():
            log_dict[sim_env_name] = {"train": {}, "test": {}}

        for sim_env_name, sim_params in self.sim_dict.items():
            sim = Task(
                self.conf.task.state_type,
                self.algo_conf.pretrained_rep,  # obj_state_type
                downsample_time_step=self.traj_info["downsample_time_step"],
                traj_scale=self.traj_info["scale"],
                goal_type=self.conf.task.goal_type,
                object_type=self.traj_info["object_type"],
                finger_type=self.traj_info["finger_type"],
                enable_shadows=sim_params["enable_shadows"],
                camera_view=sim_params["camera_view"],
                arena_color=sim_params["arena_color"],
                task=self.task,
                n_fingers_to_move=self.n_fingers_to_move,
            )
            for split_name in ["train", "test"]:
                traj_list = self.traj_info[f"{split_name}_demos"]
                plot_count_dict = {}

                totals_dict = {}
                for demo_i, demo in enumerate(traj_list):
                    diff = self.traj_info[f"{split_name}_demo_stats"][demo_i]["diff"]
                    traj_i = self.traj_info[f"{split_name}_demo_stats"][demo_i]["id"]

                    if diff in plot_count_dict:
                        if plot_count_dict[diff] >= max_demo_per_diff:
                            continue
                        else:
                            plot_count_dict[diff] += 1
                    else:
                        plot_count_dict[diff] = 1

                    log.info(
                        f"Rolling out demo (diff {diff} | id: {traj_i}) for split {split_name} in sim env {sim_env_name}"
                    )

                    traj_label = f"diff-{diff}_traj-{traj_i}"
                    traj_sim_dir = os.path.join(
                        sim_dir, sim_env_name, split_name, traj_label
                    )
                    if not os.path.exists(traj_sim_dir):
                        os.makedirs(traj_sim_dir)

                    sim_traj_dict = sim.execute_policy(
                        self.policy,
                        demo,
                        self.policy.state_dict(),
                        save_dir=traj_sim_dir,
                        encoder=self.encoder,
                        epoch=outer_i,
                    )

                    # Save gif of sim rollout
                    d_utils.save_gif(
                        sim_traj_dict["image_60"],
                        os.path.join(
                            traj_sim_dir, f"viz_{traj_label}_epoch_{outer_i+1}.gif"
                        ),
                    )

                    # Compute final error for ftpos of each finger
                    final_sim_ftpos = np.expand_dims(sim_traj_dict["ft_pos_cur"][-1], 0)
                    final_demo_ftpos = np.expand_dims(demo["ft_pos_cur"][-1], 0)
                    final_ftpos_dist = d_utils.get_per_finger_ftpos_err(
                        final_demo_ftpos, final_sim_ftpos, fnum=3
                    )
                    final_ftpos_dist = np.squeeze(final_ftpos_dist)

                    # Achieved object distance to goal
                    sim_obj_pos_err = sim_traj_dict["position_error"][-1]

                    # Compute scaled error and reward, based on task
                    scaled_reward = sim_traj_dict["scaled_success"][-1]
                    scaled_err = 1 - scaled_reward

                    # Per traj log
                    log_dict[sim_env_name][split_name][traj_label] = {
                        "sim_obj_pos_err": sim_obj_pos_err,
                        "scaled_err": scaled_err,
                        "scaled_reward": scaled_reward,
                        "final_ftpos_dist_0": final_ftpos_dist[0],
                        "final_ftpos_dist_1": final_ftpos_dist[1],
                        "final_ftpos_dist_2": final_ftpos_dist[2],
                    }

                    if diff in totals_dict:
                        totals_dict[diff]["sim_obj_pos_err"] += sim_obj_pos_err
                        totals_dict[diff]["scaled_err"] += scaled_err
                        totals_dict[diff]["scaled_reward"] += scaled_reward
                        totals_dict[diff]["final_ftpos_dist_0"] += final_ftpos_dist[0]
                        totals_dict[diff]["final_ftpos_dist_1"] += final_ftpos_dist[1]
                        totals_dict[diff]["final_ftpos_dist_2"] += final_ftpos_dist[2]
                    else:
                        totals_dict[diff] = {
                            "sim_obj_pos_err": sim_obj_pos_err,
                            "scaled_err": scaled_err,
                            "scaled_reward": scaled_reward,
                            "final_ftpos_dist_0": final_ftpos_dist[0],
                            "final_ftpos_dist_1": final_ftpos_dist[1],
                            "final_ftpos_dist_2": final_ftpos_dist[2],
                        }

                # Log avg obj pos err for each diff
                for diff, per_diff_totals_dict in totals_dict.items():
                    if (
                        f"diff-{diff}_max_avg_scaled_reward"
                        not in self.max_dict[sim_env_name][split_name].keys()
                    ):
                        self.max_dict[sim_env_name][split_name][
                            f"diff-{diff}_max_avg_scaled_reward"
                        ] = 0.0

                    for key, total in per_diff_totals_dict.items():
                        log_dict[sim_env_name][split_name][f"diff-{diff}_avg_{key}"] = (
                            total / plot_count_dict[diff]
                        )

                    curr_avg_scaled_reward = log_dict[sim_env_name][split_name][
                        f"diff-{diff}_avg_scaled_reward"
                    ]
                    if (
                        curr_avg_scaled_reward
                        > self.max_dict[sim_env_name][split_name][
                            f"diff-{diff}_max_avg_scaled_reward"
                        ]
                    ):
                        self.max_dict[sim_env_name][split_name][
                            f"diff-{diff}_max_avg_scaled_reward"
                        ] = curr_avg_scaled_reward
                        log_dict[sim_env_name][split_name][
                            f"diff-{diff}_max_avg_scaled_reward"
                        ] = curr_avg_scaled_reward
                    else:
                        log_dict[sim_env_name][split_name][
                            f"diff-{diff}_max_avg_scaled_reward"
                        ] = self.max_dict[sim_env_name][split_name][
                            f"diff-{diff}_max_avg_scaled_reward"
                        ]

            sim.close()

        return log_dict

def __del__(self):
    del self.sim
