# Copyright (c) Facebook, Inc. and its affiliates.
import random
import os
import sys
import torch
import numpy as np
import higher

# import matplotlib.pyplot as plt
import argparse
import collections
import wandb
from r3m import load_r3m
import torchvision.transforms as T
from PIL import Image

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, ".."))

import utils.data_utils as d_utils

# Set run logging directory to be trifinger_mbirl
mbirl_dir = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(mbirl_dir, "logs/runs")


class ImitationLearningDataset(torch.utils.data.Dataset):
    def __init__(self, demos, device="cpu"):
        self.dataset = []
        transforms = T.Compose(
            [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
        )  # ToTensor() divides by 255

        for demo in demos:
            num_obs = demo["o_pos_cur"].shape[0]

            for i in range(num_obs):

                image = demo["image_60"][i]
                goal_image = demo["image_60"][-1]
                image_preproc = transforms(
                    Image.fromarray(image.astype(np.uint8))
                ).reshape(3, 224, 224)
                goal_image_preproc = transforms(
                    Image.fromarray(goal_image.astype(np.uint8))
                ).reshape(3, 224, 224)
                obs_dict = {
                    "o_pos_cur": demo["o_pos_cur"][i],
                    "ft_pos_cur": demo["ft_pos_cur"][i],
                    "o_pos_des": demo["o_pos_des"][0, :],  # Goal object position
                    "image_60": image_preproc,
                    "image_60_goal": goal_image_preproc,
                }

                action = torch.FloatTensor(demo["delta_ftpos"][i]).to(device)

                self.dataset.append((obs_dict, action))

        # TODO make obs relative to goal (final, and intermmediate)

        # self.dataset = [
        #     (torch.FloatTensor(x[i]), torch.FloatTensor(y[i])) for i in range(len(x))
        # ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def plot_loss(loss_dict, outer_i):
    log_dict = {f"{k}": v for k, v in loss_dict.items()}
    log_dict["outer_i"] = outer_i
    wandb.log(log_dict)


def train(conf, dataloader, policy, model_data_dir):

    # Make logging directories
    ckpts_dir = os.path.join(model_data_dir, "ckpts")
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)

    r3m = load_r3m("resnet50")  # resnet18, resnet34
    r3m.eval()
    r3m.to("cpu")

    bc_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(policy.parameters(), lr=conf.lr)

    for outer_i in range(conf.n_outer_iter):
        for batch, (obs_dict, actions) in enumerate(dataloader):
            image = obs_dict["image_60"]
            visual_obs = r3m(image * 255.0)[0]
            proprio_obs = torch.FloatTensor(obs_dict["ft_pos_cur"]).to("cpu")

            # Goal image
            image_goal = obs_dict["image_60_goal"]
            visual_obs_goal = r3m(image_goal * 255.0)[0]

            obs = torch.cat([visual_obs, proprio_obs, visual_obs_goal])

            optimizer.zero_grad()
            pred_actions = policy(obs)
            loss = bc_loss(pred_actions, actions)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {outer_i}, loss: {loss.item()}")

        if (outer_i + 1) % conf.n_epoch_every_log == 0:

            torch.save(
                {
                    "bc_loss_train": loss,
                    "policy": policy.state_dict(),
                    "conf": conf,
                },
                f=f"{ckpts_dir}/epoch_{outer_i+1}_ckpt.pth",
            )
