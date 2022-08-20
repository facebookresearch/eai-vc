import torch
import numpy as np

# Model
class ForwardModel(torch.nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dims):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.activation = torch.nn.ReLU(inplace=True)
        #self.activation = torch.nn.Tanh()

        dims = [in_dim] + list(hidden_dims) + [out_dim]
        module_list = []
        for i in range(len(dims) - 1):
            module_list.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                module_list.append(self.activation) 

        self.model_list = torch.nn.ModuleList(module_list)

    def forward(self, x):
            
        for i in range(len(self.model_list)):
            x = self.model_list[i](x)

        return x

def get_obs_vec_from_obs_dict(obs_dict, use_ftpos=True):
    """
    Concatenate states and action from obs_dict, choosing whether or not to use ftpos
    obs_dict values can be batched, with batch size B

    args:
        - obs_dict: {
            "ft_state": fingertip positions [B, 9],
            "o_state": object state [B, o_dim],
            "actions": actions [B, 9]
        }
        - use_ftpos: If True, include ft_state in obs
    """

    if use_ftpos:
        obs = torch.cat([obs_dict["ft_state"], obs_dict["o_state"], obs_dict["action"]], dim=1)
    else:
        obs = torch.cat([obs_dict["o_state"], obs_dict["action"]], dim=1)

    return obs
