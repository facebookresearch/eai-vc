# Copyright (c) Facebook, Inc. and its affiliates.
import random
import os
import sys
import torch
import numpy as np
import higher
#import matplotlib.pyplot as plt
import argparse
import collections
import wandb
from r3m import load_r3m
import torchvision.transforms as T
from PIL import Image

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

import utils.data_utils as d_utils

# Set run logging directory to be trifinger_mbirl
mbirl_dir = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(mbirl_dir, "logs/runs")


class ImitationLearningDataset(torch.utils.data.Dataset):
    def __init__(self, demos, obs_type="goal_none", device="cpu"):
        self.dataset = []

        r3m = load_r3m("resnet50")  # resnet18, resnet34
        r3m.eval()
        r3m.to(device)

        for demo in demos:
            num_obs = demo['o_pos_cur'].shape[0]

            for i in range(num_obs):
                obs_dict = {
                            "o_pos_cur" : demo["o_pos_cur"][i],
                            "ft_pos_cur": demo["ft_pos_cur"][i],
                            "o_pos_des" : demo["o_pos_des"][0, :], # Goal object position
                            "image_60"  : demo["image_60"][i],
                           }

                obs = get_bc_obs(obs_dict, obs_type, r3m=r3m, device=device)

                action = torch.FloatTensor(demo['delta_ftpos'][i]).to(device)

                self.dataset.append((obs, action))

        # TODO make obs relative to goal (final, and intermmediate)

        # self.dataset = [
        #     (torch.FloatTensor(x[i]), torch.FloatTensor(y[i])) for i in range(len(x))
        # ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_bc_obs(obs_dict, obs_type, r3m=None, device="cpu"):
    """
    Return obs for policy for given obs_type

    args:
        obs_dict (dict): 
        obs_type (str): [
                         "goal_none", # fingertip pos and object pos in world frame, no goal [12]
                         "goal_rel",  # fingertip pos and object pos, relative to object goal [12]
                         "img_r3m",   # image passed through pretrained r3m + fingertip pos [2048 + 9]
                        ]
    """

    if obs_type == "goal_none":
        obs = torch.cat([torch.FloatTensor(obs_dict["o_pos_cur"]), torch.FloatTensor(obs_dict["ft_pos_cur"])])

    elif obs_type == "goal_rel":
        ft_pos_rel = np.repeat(obs_dict["o_pos_des"], 3) - obs_dict["ft_pos_cur"]
        o_pos_rel  = obs_dict["o_pos_des"] - obs_dict["o_pos_cur"]

        obs = torch.cat([torch.FloatTensor(ft_pos_rel), torch.FloatTensor(o_pos_rel)])

    elif obs_type == "img_r3m":
        transforms = T.Compose([T.Resize(256),
                     T.CenterCrop(224),
                     T.ToTensor()]) # ToTensor() divides by 255

        image = obs_dict["image_60"]
        image_preproc = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
        visual_obs = r3m(image_preproc * 255.0)[0].detach()
        proprio_obs = torch.FloatTensor(obs_dict["ft_pos_cur"]).to(device)
        obs = torch.cat([visual_obs, proprio_obs])

    else:
        raise ValueError("Invalid obs_type")

    return obs

def plot_loss(loss_dict, outer_i):
    log_dict = {f'{k}': v for k, v in loss_dict.items()}
    log_dict['outer_i'] = outer_i
    wandb.log(log_dict)

def train(conf, dataloader, policy, model_data_dir):

    # Make logging directories
    ckpts_dir = os.path.join(model_data_dir, "ckpts") 
    if not os.path.exists(ckpts_dir): os.makedirs(ckpts_dir)

    bc_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(policy.parameters(), lr=conf.bc_lr)

    for outer_i in range(conf.n_outer_iter):
        for batch, (obs, actions) in enumerate(dataloader):
            optimizer.zero_grad()
            pred_actions = policy(obs)
            loss = bc_loss(pred_actions, actions)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {outer_i}, loss: {loss.item()}")

        if (outer_i+1) % conf.n_epoch_every_log == 0:

            torch.save({
                'bc_loss_train' : loss,
                'policy'        : policy.state_dict(),
                'conf'          : conf,
            }, f=f'{ckpts_dir}/epoch_{outer_i+1}_ckpt.pth')

       

