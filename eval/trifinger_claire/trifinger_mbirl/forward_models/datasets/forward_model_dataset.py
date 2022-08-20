import torch
import numpy as np
import sys
import os

class ForwardModelDataset(torch.utils.data.Dataset):
    def __init__(self, demo_list, obj_state_type="vertices", device="cpu"):
        """
        demo_list: List of demo dicts
        obj_state_type (str): "pos" or "vertices"
        """
        
        assert obj_state_type in ["pos", "vertices", "img_r3m"]
        self.obj_state_type = obj_state_type
    
        # Make dataset from demo list, and save
        self.dataset = self.make_dataset_from_demo_list(demo_list)

        # Save dataset as json
        data_info_dict = {"dataset": self.dataset}

        # Dimensions
        self.a_dim = self.dataset[0]["obs"]["action"].shape[0]
        self.o_state_dim = self.dataset[0]["obs"]["o_state"].shape[0]
        self.ft_state_dim = self.dataset[0]["obs"]["ft_state"].shape[0]
        
        #variance = self.get_target_variance()
        #print(variance)

    def make_dataset_from_demo_list(self, demos):
        """ """

        dataset = []

        for demo in demos:

            num_obs = demo['o_pos_cur'].shape[0]

            for i in (range(num_obs-1)): # TODO train on full trajectories
                # Object positions
                o_pos_cur = demo["o_pos_cur"][i]
                o_pos_next = demo["o_pos_cur"][i+1]

                # Object vertices
                o_vert_cur = demo["vertices"][i]
                o_vert_next = demo["vertices"][i+1]

                # Current fingertip positions
                ft_pos_cur = demo["ft_pos_cur"][i]
                ft_pos_next = demo["ft_pos_cur"][i+1]

                # Action (fingertip position deltas)
                action = torch.FloatTensor(demo['delta_ftpos'][i])

                # Make state and action
                if self.obj_state_type == "pos":
                    o_state_cur = torch.FloatTensor(o_pos_cur)
                    o_state_next = torch.FloatTensor(o_pos_next)

                elif self.obj_state_type == "vertices":
                    o_state_cur = torch.FloatTensor(o_vert_cur)
                    o_state_next = torch.FloatTensor(o_vert_next)

                elif self.obj_state_type == "img_r3m":
                    o_state_cur = torch.FloatTensor(demo["image_60_r3m"][i])
                    o_state_next = torch.FloatTensor(demo["image_60_r3m"][i+1])

                else:
                    raise ValueError("Invalid obj_state_type")    
            
                # Observation dict (current state and action)
                obs_dict = {"ft_state": torch.FloatTensor(ft_pos_cur),
                            "o_state": o_state_cur, 
                            "action": torch.FloatTensor(action)
                }

                # Next state dict
                state_next_dict = {
                    "ft_state": torch.FloatTensor(ft_pos_next),
                    "o_state": o_state_next
                }

                data_dict = {
                             "obs": obs_dict,
                             "state_next": state_next_dict, 
                            }

                dataset.append(data_dict)

        return dataset


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    # TODO need to update for state_next_dict format
    #def get_target_variance(self):
    #    state_next = torch.stack([self.dataset[i]["state_next"] for i in range(len(self.dataset))])
    #    return state_next.var(dim=0)
