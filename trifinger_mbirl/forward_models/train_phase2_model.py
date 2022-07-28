import torch
import numpy as np
import sys
import os
import argparse
import random
import collections

from r3m import load_r3m
import torchvision.transforms as T
from PIL import Image

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))
sys.path.insert(0, os.path.join(base_path, '../..'))

import utils.data_utils as d_utils

# Set run logging directory to be trifinger_mbirl
models_dir = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(models_dir, "runs")

"""
Train phase 2 model
"""

class NMSELoss(torch.nn.Module):
    def __init__(self, target_vars):
        self.target_vars = target_vars
        super(NMSELoss, self).__init__()

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in - y_target) ** 2).mean(dim=0)

        assert mse.shape[-1] == y_in.shape[-1]
        nmse = mse / self.target_vars

        return nmse.mean()

## Dataset
class Phase2ModelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, demo_list=None, obj_state_type="vertices", device="cpu"):
        """
        demo_list: List of demo dicts
        obj_state_type (str): "pos" or "vertices"
        """
        
        ### ONLY TAKE PHASE 2 PART OF TRAJECTORY

        assert obj_state_type in ["pos", "vertices", "img_r3m"]
        self.obj_state_type = obj_state_type
    
        if demo_list is not None:
            # Make dataset from demo list, and save
            self.dataset, self.phase2_start_ind = self.make_dataset_from_demo_list(demo_list)

            # Save dataset as json
            data_info_dict = {"dataset": self.dataset, "phase2_start_ind": self.phase2_start_ind}
            
            if dataset_path is not None:
                demo_dir = os.path.dirname(dataset_path)
                if not os.path.exists(demo_dir): os.makedirs(demo_dir)
                torch.save(data_info_dict, f=dataset_path)
                print(f"Saved dataset to {dataset_path}")
        else:
            if os.path.exists(dataset_path):
                print(f"Loading dataset from {dataset_path}")
                # Load dataset from json
                data_info_dict = torch.load(dataset_path)
                self.dataset = data_info_dict["dataset"]
                self.phase2_start_ind = data_info_dict["phase2_start_ind"]
            else:
                raise ValueError(f"{dataset_path} does not exist")

        # Obs dimensions
        self.in_dim = 0
        obs_dict = self.dataset[0]["obs"]
        for k, v in obs_dict.items():
            self.in_dim += v.shape[0]

        # Next state dimensions
        self.out_dim = self.dataset[0]["state_next"].shape[0]

        variance = self.get_target_variance()
        print(variance)

    def make_dataset_from_demo_list(self, demos):
        """ """

        dataset = []

        for demo in demos:

            num_obs = demo['o_pos_cur'].shape[0]

            # Find phase2 start index in trajectory
            phase2_start_ind = demo["mode"].tolist().index(2)

            #for i in (range(phase2_start_ind, num_obs-1)):
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

                dataset.append(data_dict)

        return dataset, phase2_start_ind


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_target_variance(self):
        state_next = torch.stack([self.dataset[i]["state_next"] for i in range(len(self.dataset))])
        return state_next.var(dim=0)


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
        self.activation = torch.nn.ReLU(inplace=True)
        #self.activation = torch.nn.Tanh()

        dims = [in_dim] + list(hidden_dims) + [out_dim]
        module_list = []
        for i in range(len(dims) - 1):
            module_list.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                module_list.append(self.activation) 

        self.model_list = torch.nn.ModuleList(module_list)

    def forward(self, obs_dict):
            
        x = torch.cat([obs_dict["ft_state"], obs_dict["o_state"], obs_dict["action"]], dim=1)
        for i in range(len(self.model_list)):
            x = self.model_list[i](x)

        return x


def train(conf, model, loss_fn, optimizer, dataloader, test_dataloader, phase2_start_ind, model_data_dir=None):

    for epoch in range(conf.n_epochs):
        model.train()
        avg_loss = 0.0
        for i_batch, batch in enumerate(dataloader):
            optimizer.zero_grad()
            pred_state_next = model(batch["obs"])
            loss = loss_fn(pred_state_next, batch["state_next"])
            loss = loss.mean()
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
                    test_loss_mean = test_loss.mean(dim=0)
                    print(test_loss_mean / torch.max(test_loss_mean)) 
                    test_loss = test_loss.mean()
                    print(f"Epoch: {epoch}, test loss: {test_loss.item()}")

        if (epoch+1) % 500 == 0:
            torch.save({
                'loss_train'       : loss,
                'model_state_dict' : model.state_dict(),
                'conf'             : conf,
                'in_dim'           : model.in_dim,
                'out_dim'          : model.out_dim,
                'hidden_dims'      : model.hidden_dims,
                'phase2_start_ind' : phase2_start_ind,
            }, f=f'{model_data_dir}/epoch_{epoch+1}_ckpt.pth')

def get_exp_str(params_dict):
    
    sorted_dict = collections.OrderedDict(sorted(params_dict.items()))

    exp_str = f"phase2_model"

    if params_dict["file_path"] is not None:
        file_path = os.path.splitext(os.path.split(params_dict["file_path"])[1])[0]
        exp_str += f"_{file_path}"

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

    DEMO_DIR = "/Users/clairelchen/logs/demos/"
    SCALE = 100
    if args.file_path is not None:
        train_trajs, test_trajs = d_utils.load_trajs(args.file_path, scale=SCALE)
        train_data_path = None
        test_data_path = None
    else:
        TEST_DEMO_ID = 5 # ID of test trajectory
        dataset_dir = os.path.join(DEMO_DIR, "datasets")
        train_data_path = os.path.join(dataset_dir, f"dataset_train-{args.n_train}_ost-{args.obj_state_type}_s-{SCALE}.pth")
        test_data_path = os.path.join(dataset_dir, f"dataset_test_id-{TEST_DEMO_ID}_ost-{args.obj_state_type}_s-{SCALE}.pth")

        if os.path.exists(train_data_path) and os.path.exists(test_data_path):
            # Dataset already saved, save time by not loading demo info
            train_trajs, test_trajs = None, None
        else:
            # Load demo info for making dataset
            train_demos = list(range(args.n_train+1))
            train_demos.remove(TEST_DEMO_ID)
            traj_info = {
                    "demo_dir"   : DEMO_DIR,
                    "difficulty" : 1,
                    "train_demos": train_demos,
                    "test_demos" : [TEST_DEMO_ID]
                }

            # Load train and test trajectories
            train_trajs, test_trajs = d_utils.load_trajs(traj_info, scale=SCALE)

    # Name experiment and make exp directory
    exp_str = get_exp_str(vars(conf))
    exp_dir = os.path.join(conf.log_dir, exp_str)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Datasets and dataloaders
    train_dataset = Phase2ModelDataset(train_data_path, demo_list=train_trajs,
                                       obj_state_type=conf.obj_state_type, device=device)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = Phase2ModelDataset(test_data_path, demo_list=test_trajs,
                                      obj_state_type=conf.obj_state_type, device=device)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    in_dim = train_dataset.in_dim
    out_dim = train_dataset.out_dim
    hidden_dims = [100]

    model = Phase2Model(in_dim, out_dim, hidden_dims)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    #target_var = train_dataset.get_target_variance()
    #loss_fn = NMSELoss(target_vars=target_var)
    loss_fn = torch.nn.MSELoss(reduction="none")

    train(conf, model, loss_fn, optimizer, dataloader, test_dataloader, train_dataset.phase2_start_ind, model_data_dir=exp_dir)

def parse_args():

    parser = argparse.ArgumentParser()

    # Required for specifying training and test trajectories
    parser.add_argument("--file_path", default=None, help="""Filepath of trajectory to load""")
    # OR
    parser.add_argument("--n_train", type=int, default=20, help="Number of training trajectories")

    parser.add_argument("--n_epochs", type=int, default=1500, help="Number of epochs")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory for run logs")

    parser.add_argument("--obj_state_type", type=str, default="pos", choices=["pos", "vertices", "img_r3m"],
                        help="Object state representation")

    parser.add_argument("--n_epoch_every_eval", type=int, default=100, help="Num epochs every eval")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

