import torch
import numpy as np
import sys
import os
import argparse
import random
import collections

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))
sys.path.insert(0, os.path.join(base_path, '../..'))

import utils.data_utils as d_utils

# Set run logging directory to be trifinger_mbirl
models_dir = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(models_dir, "runs")

## Dataset
class Phase2ModelDataset(torch.utils.data.Dataset):
    def __init__(self, demos, obj_state_type="vertices"):
        """
        demos: List of demo dicts
        obj_state_type (str): "pos" or "vertices"
        """
        
        ### ONLY TAKE PHASE 2 PART OF TRAJECTORY

        assert obj_state_type in ["pos", "vertices"]

        self.dataset = []

        for demo in demos:

            num_obs = demo['o_pos_cur'].shape[0]

            # Find phase2 start index in trajectory
            phase2_start_ind = demo["mode"].tolist().index(2)

            for i in (range(phase2_start_ind, num_obs-1)):
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
                if obj_state_type == "pos":
                    o_state_cur = torch.FloatTensor(o_pos_cur)
                    o_state_next = torch.FloatTensor(o_pos_next)

                elif obj_state_type == "vertices":
                    o_state_cur = torch.FloatTensor(o_vert_cur)
                    o_state_next = torch.FloatTensor(o_vert_next)

                else:
                    raise ValueError("Invalid obj_state_type")    
            
                # Append action to observation
                obs_dict = {"ft_state": torch.FloatTensor(ft_pos_cur),
                            "o_state": o_state_cur, 
                            "action": torch.FloatTensor(action)
                            }

                # Concate next ft state and next obj state
                state_next = torch.cat([torch.FloatTensor(ft_pos_next), o_state_next])

                data_dict = {
                             "obs": obs_dict,
                             "state_next": state_next, 
                            }

                self.dataset.append(data_dict)

                # Obs dimensions
                self.in_dim = 0
                for k, v in obs_dict.items():
                    self.in_dim += v.shape[0]

                # Next state dimensions
                self.out_dim = data_dict["state_next"].shape[0]


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

## Model
class Phase2Model(torch.nn.Module):

    """
    Input: current object state, current fingertip positions, fingertip deltas (action)
    Output: next object state
    """

    def __init__(self, in_dim, out_dim, hidden_dims):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        #self.activation = torch.nn.ReLU(inplace=True)
        self.activation = torch.nn.Tanh()

        dims = [in_dim] + list(hidden_dims) + [out_dim]
        module_list = []
        for i in range(len(dims) - 1):
            module_list.append(torch.nn.Linear(dims[i], dims[i+1]))
            module_list.append(self.activation) 

        self.model_list = torch.nn.ModuleList(module_list)

    def forward(self, obs_dict):
            
        x = torch.cat([obs_dict["ft_state"], obs_dict["o_state"], obs_dict["action"]], dim=1)
        for i in range(len(self.model_list)):
            x = self.model_list[i](x)

        return x


def train(conf, model, loss_fn, optimizer, dataloader, test_dataloader, model_data_dir=None):

    for epoch in range(conf.n_epochs):
        model.train()
        avg_loss = 0.0
        for i_batch, batch in enumerate(dataloader):
            optimizer.zero_grad()
            pred_state_next = model(batch["obs"])
            loss = loss_fn(pred_state_next, batch["state_next"])
            loss.backward()
            optimizer.step()
        
            avg_loss += loss.item()

        print(f"Epoch: {epoch}, loss: {avg_loss/(i_batch+1)}")

        if (epoch+1) % conf.n_epoch_every_eval == 0:
            model.eval()
            with torch.no_grad():
                for i_batch, batch in enumerate(test_dataloader):
                    pred_state_next = model(batch["obs"])
                    test_loss = loss_fn(pred_state_next, batch["state_next"])
                    print(f"Epoch: {epoch}, test loss: {test_loss.item()}")

    torch.save({
        'loss_train'       : loss,
        'model_state_dict' : model.state_dict(),
        'conf'             : conf,
        'in_dim'           : model.in_dim,
        'out_dim'          : model.out_dim,
        'hidden_dims'      : model.hidden_dims,
    }, f=f'{model_data_dir}/epoch_{epoch+1}_ckpt.pth')

def get_exp_str(params_dict):
    
    sorted_dict = collections.OrderedDict(sorted(params_dict.items()))

    file_path = os.path.splitext(os.path.split(params_dict["file_path"])[1])[0]
 
    exp_str = f"phase2_model_{file_path}"

    for key, val in sorted_dict.items():
        # exclude these keys from exp name
        if key in ["file_path", "no_wandb", "log_dir", "run_id", "n_epoch_every_eval", "n_epochs"]: continue

        # Abbreviate key
        splits = key.split("_")
        short_key = ""
        for split in splits:
            short_key += split[0]
    
        exp_str += "_{}-{}".format(short_key, str(val).replace(".", "p"))

    return exp_str
            
    
## Trainining
def main(conf):
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load train and test trajectories
    train_trajs, test_trajs = d_utils.load_trajs(args.file_path)

    # Name experiment and make exp directory
    exp_str = get_exp_str(vars(conf))
    exp_dir = os.path.join(conf.log_dir, exp_str)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Train dataloader
    train_dataset = Phase2ModelDataset(train_trajs, obj_state_type=conf.obj_state_type)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = Phase2ModelDataset(test_trajs, obj_state_type=conf.obj_state_type)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    in_dim = train_dataset.in_dim
    out_dim = train_dataset.out_dim
    hidden_dims = [100, 25]

    model = Phase2Model(in_dim, out_dim, hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    train(conf, model, loss_fn, optimizer, dataloader, test_dataloader,  model_data_dir=exp_dir)

def parse_args():

    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--file_path", default=None, help="""Filepath of trajectory to load""")

    parser.add_argument("--n_epochs", type=int, default=1500, help="Number of epochs")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory for run logs")

    parser.add_argument("--obj_state_type", type=str, default="pos", choices=["pos", "vertices"],
                        help="Object state representation")

    parser.add_argument("--n_epoch_every_eval", type=int, default=100, help="Num epochs every eval")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

