# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from trifinger_vc.utils.model_utils import MODEL_NAMES, get_vc_model_and_transform
import vc_models.transforms as vc_t


class BCFinetuneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        demo_list,
        state_type="ftpos_obj",
        obj_state_type="pos",
        device="cpu",
        augment_prob=0.0,
        times_to_use_demo=1,
        jitter_brightness=0.5,
        jitter_contrast=0.5,
        jitter_saturation=0.5,
        jitter_hue=0.03,
        shift_pad=10,
        fingers_to_move=3,
        task="move_cube",
        demo_root_dir="assets/data/trifinger-demos",
    ):
        """
        args:
            state_type: ["ftpos", "ftpos_obj", "obj"]
            obj_state_type: ["pos", "vertices"] +  MODEL_NAMES
            augment_prob: probablity to augment images [0.0, 1.0]
            times_to_use_demo: (int) number of times to use each demo
            task: task name (str)
                if "reach_cube", take subset of action corresponding to diff (number of fingers that move)
        """
        self.dataset = []
        self.state_type = state_type
        self.obj_state_type = obj_state_type
        self.device = device
        self.augment_prob = augment_prob
        self.times_to_use_demo = times_to_use_demo
        self.n_fingers_to_move = fingers_to_move

        # Get transformation for img preproc
        if self.obj_state_type in MODEL_NAMES:
            _, self.preproc_transform, pretrained_rep_dim = get_vc_model_and_transform(
                self.obj_state_type, device=self.device
            )
        else:
            raise NameError

        # Make dataset from demo list, and save
        self.dataset = []

        self.n_augmented_samples = 0

        # Random image shift and color jitter
        self.rand_augment_transforms = vc_t.transform_augment(
            # Resize/crop
            resize_size=256,
            output_size=224,
            # Jitter
            jitter=True,
            jitter_prob=1.0,
            jitter_brightness=jitter_brightness,
            jitter_contrast=jitter_contrast,
            jitter_saturation=jitter_saturation,
            jitter_hue=jitter_hue,
            # Shift
            shift=True,
            shift_pad=shift_pad,
            # Randomize environments
            randomize_environments=False,
        )

        for demo_stats in demo_list:
            if demo_root_dir is not None:
                demo_dir = os.path.join(demo_root_dir,demo_stats["path"])
            else:
                demo_dir = demo_stats["path"]
            self.add_new_traj(demo_dir)

        # Dimensions
        self.out_dim = self.dataset[0]["output"]["action"].shape[0]
        self.pretrained_rep_dim = pretrained_rep_dim

    def add_new_traj(self, demo_dir):
        # Read data from demo_dir

        downsample_data_path = os.path.join(demo_dir, "downsample.pth")
        if not os.path.exists(downsample_data_path):
            print(f"{downsample_data_path} not found")
            return
        demo = torch.load(downsample_data_path)

        num_obs = demo["o_pos_cur"].shape[0]

        # Goal position (absolute)
        o_goal_pos = torch.FloatTensor(demo["o_pos_cur"][-1]).to(self.device)

        # Goal position (relative)
        o_init_pos = torch.FloatTensor(demo["o_pos_cur"][0]).to(self.device)
        o_goal_pos_rel = o_goal_pos - o_init_pos

        # Goal image
        orig_img_goal = (
            torch.Tensor(demo["image_60"][-1]).permute((2, 0, 1)) / 255.0
        )  # [3, 270, 270]

        for i in range(num_obs - 1):
            # Current fingertip positions
            ft_pos_cur = demo["ft_pos_cur"][i]

            # Action (fingertip position deltas)
            action = torch.FloatTensor(demo["delta_ftpos"][i])
            # Get subset of delta_ftpos that corresonds to diff (number of fingers that move)
            # For the reach task this will be [:3], and for other tasks [:9]
            action = action[: self.n_fingers_to_move * 3]

            # transform expects images as float tensor with values in range [0.0, 1.0]
            orig_img = (
                torch.Tensor(demo["image_60"][i]).permute((2, 0, 1)) / 255.0
            )  # [3, 270, 270]

            for j in range(self.times_to_use_demo):
                # Augment images
                if np.random.rand() < self.augment_prob:
                    img = self.rand_augment_transforms(orig_img)
                    self.n_augmented_samples += 1
                else:
                    img = orig_img

                # For testing
                # plt.imsave(f"test_img_{i}_aug_{j}.png", img.permute(1,2,0).detach().numpy())

                # Transform images for encoder
                img_preproc = self.preproc_transform(img).to(self.device)
                img_preproc_goal = self.preproc_transform(orig_img_goal).to(self.device)

                # Observation dict (current state and action)
                input_dict = {
                    "ft_state": torch.FloatTensor(ft_pos_cur).to(self.device),
                    "rgb_img_preproc": img_preproc,
                    "rgb_img_preproc_goal": img_preproc_goal,
                    "o_goal_pos": o_goal_pos,
                    "o_goal_pos_rel": o_goal_pos_rel,
                }

                output_dict = {
                    "action": torch.FloatTensor(action).to(self.device),
                }

                data_dict = {"input": input_dict, "output": output_dict}

                self.dataset.append(data_dict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # TODO instead of reading all demos into memory, can read from files each time here
        # and apply image augmentation

        return self.dataset[idx]
