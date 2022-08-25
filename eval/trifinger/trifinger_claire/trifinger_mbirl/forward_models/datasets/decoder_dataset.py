import torch
import numpy as np
import sys
import os

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, ".."))

import utils.data_utils as d_utils


class DecoderDataset(torch.utils.data.Dataset):
    def __init__(self, demo_list, demo_stats, device="cpu"):
        """
        demo_list: List of demo dicts
        """

        # Make dataset from demo list, and save
        self.dataset = []

        for demo_i, demo in enumerate(demo_list):

            num_obs = demo["o_pos_cur"].shape[0]

            for i in range(num_obs):
                img = demo["image_60"][i]
                resized_img = d_utils.resize_img(img).to(device)  # [3,64,64]
                r3m_vec = torch.Tensor(demo["image_60_r3m"][i]).to(device)

                data_dict = {
                    "demo_stats": demo_stats[demo_i],
                    "r3m_vec": r3m_vec,
                    "rgb_img": resized_img,
                }

                self.dataset.append(data_dict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
