import numpy as np
import torch


class Phase1Model(torch.nn.Module):
    def forward(self, obs_dict):

        ft_pos = obs_dict["ft_state"]
        o_state = obs_dict["o_state"]
        action = obs_dict["action"]

        ft_pos_next = ft_pos + action
        o_state_next = o_state

        # TODO concat ft_pos and o_state
        state_next = torch.cat([ft_pos_next, o_state_next], dim=1)
        return state_next
