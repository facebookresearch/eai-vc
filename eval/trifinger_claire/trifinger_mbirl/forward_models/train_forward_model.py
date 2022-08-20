import torch
import numpy as np
import sys
import os
import argparse
import random
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import logging

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))
sys.path.insert(0, os.path.join(base_path, '../..'))

from trifinger_mbirl.forward_models.datasets.forward_model_dataset import ForwardModelDataset
from trifinger_mbirl.forward_models.models.forward_model import ForwardModel, get_obs_vec_from_obs_dict
from trifinger_mbirl.forward_models.models.decoder_model import DecoderModel
from trifinger_mbirl.learned_mpc import LearnedMPC, test_mpc
import utils.data_utils as d_utils
import utils.train_utils as t_utils

# A logger for this file
log = logging.getLogger(__name__)

"""
Train forward model, with default config params:
python trifinger_mbirl/forward_model/train_forward_model.py algo=forward_model

Will save outputs to trifinger_mbirl/outputs

Note: For now, you need to run the script from the trifinger_claire/ directory, because 
of the way the demo_path param in defined in config.yaml
"""

def get_obs_and_state_next_from_batch(batch, use_ftpos=True):

    obs = get_obs_vec_from_obs_dict(batch["obs"], use_ftpos=use_ftpos)

    if use_ftpos:
        gt_state_next = torch.cat([batch["state_next"]["ft_state"], batch["state_next"]["o_state"]], dim=1)
    else:
        gt_state_next = batch["state_next"]["o_state"]

    return obs, gt_state_next

def train(conf, model, loss_fn, optimizer, train_dataloader, test_dataloader, traj_info,
        model_data_dir=None):

    ckpts_dir = os.path.join(model_data_dir, "ckpts") 
    plots_dir = os.path.join(model_data_dir, "plots") 
    if not os.path.exists(ckpts_dir): os.makedirs(ckpts_dir)
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)

    # Load and use decoder to viz pred_o_states
    if conf.algo.path_to_decoder_ckpt is not None:
        decoder_model_dict = torch.load(conf.algo.path_to_decoder_ckpt) 
        decoder = DecoderModel()
        decoder.load_state_dict(decoder_model_dict["model_state_dict"])
    else:
        decoder = None

    for epoch in range(conf.algo.n_epochs):
        # Unfreeze network params (params are frozen during mpc rollout)
        for name, param in model.named_parameters():
            param.requires_grad = True

        # Train
        model.train()
        total_train_loss = 0.0
        for i_batch, batch in enumerate(train_dataloader):

            obs, gt_state_next = get_obs_and_state_next_from_batch(batch, use_ftpos=conf.algo.use_ftpos)

            optimizer.zero_grad()
            pred_state_next = model(obs)
            loss = loss_fn(pred_state_next, gt_state_next)
            loss.backward()
            optimizer.step()
        
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss/(i_batch+1)
        print(f"Epoch: {epoch}, loss: {avg_train_loss}")

        # Get loss on test set
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for i_batch, batch in enumerate(test_dataloader):
                obs, gt_state_next = get_obs_and_state_next_from_batch(batch, use_ftpos=conf.algo.use_ftpos)
                pred_state_next = model(obs)
                test_loss = loss_fn(pred_state_next, gt_state_next)
                total_test_loss += test_loss.item()
        avg_test_loss = total_test_loss/(i_batch+1)
        print(f"Epoch: {epoch}, test loss: {avg_test_loss}")

        loss_dict = {
                    "train_loss": avg_train_loss, 
                    "test_loss": avg_test_loss, 
                    }

        if (epoch+1) % conf.algo.n_epoch_every_save == 0:
            # Save checkpoint
            model_dict = {
                'loss_train'       : loss,
                'model_state_dict' : model.state_dict(),
                'conf'             : conf,
                'in_dim'           : model.in_dim,
                'out_dim'          : model.out_dim,
                'hidden_dims'      : model.hidden_dims,
            }
            torch.save(model_dict, f=f'{ckpts_dir}/epoch_{epoch+1}_ckpt.pth')

            # Eval model in rollout
            time_horizon = traj_info["train_demos"][0]["ft_pos_cur"].shape[0]
            # Make MPC
            mpc = LearnedMPC(time_horizon-1, model_dict=model_dict)

            MAX_PLOT_PER_DIFF = 10
            for split_name in ["train", "test"]:
                traj_list = traj_info[f"{split_name}_demos"]

                for one_step in [False, True]:
                    save_label = "pred_one_step" if one_step else "pred_rollout"
                    max_ft_pos_l2_dist, max_o_state_l2_dist = -np.inf, -np.inf
                    avg_ft_pos_l2_dist, avg_o_state_l2_dist = 0, 0

                    plot_count_dict = {}
                    for i, traj in enumerate(traj_list):
                        # Make save directory for plots
                        diff = traj_info[f"{split_name}_demo_stats"][i]["diff"]
                        traj_i = traj_info[f"{split_name}_demo_stats"][i]["id"]
                        
                        if diff in plot_count_dict:
                            if plot_count_dict[diff] >= MAX_PLOT_PER_DIFF: continue
                            else: plot_count_dict[diff] += 1
                        else:
                            plot_count_dict[diff] = 1

                        traj_dir = os.path.join(plots_dir, split_name, f"diff-{diff}_traj-{traj_i}")
                        if not os.path.exists(traj_dir): os.makedirs(traj_dir)

                        # Run MPC
                        pred_traj_dict, pred_err_dict = test_mpc(mpc, traj, epoch+1,
                                                                 save_dir=traj_dir, one_step=one_step)

                        max_ft_pos_l2_dist = max(pred_err_dict["max_ft_pos_l2_dist"], max_ft_pos_l2_dist)
                        max_o_state_l2_dist = max(pred_err_dict["max_o_state_l2_dist"], max_o_state_l2_dist)
                        avg_ft_pos_l2_dist += pred_err_dict["avg_ft_pos_l2_dist"]
                        avg_o_state_l2_dist += pred_err_dict["avg_o_state_l2_dist"]

                        if decoder is not None:
                            pred_imgs = decoder(torch.Tensor(pred_traj_dict["o_state"]))
                            decoder.save_gif(pred_imgs, os.path.join(traj_dir, f'{save_label}_epoch_{epoch+1}.gif'))

                    loss_dict[f"{split_name}_{save_label}_max_ft_pos_l2_dist"] = max_ft_pos_l2_dist
                    loss_dict[f"{split_name}_{save_label}_max_o_state_l2_dist"] = max_o_state_l2_dist
                    loss_dict[f"{split_name}_{save_label}_avg_ft_pos_l2_dist"] = avg_ft_pos_l2_dist / len(traj_list)
                    loss_dict[f"{split_name}_{save_label}_avg_o_state_l2_dist"] = avg_o_state_l2_dist / len(traj_list)

            del mpc

        print(loss_dict)
        if not conf.no_wandb:
            t_utils.plot_loss(loss_dict, epoch+1)


## Training
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(conf):
    assert conf.algo.name == "forward_model", "Need to use algo=forward_model when running script"

    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Name experiment and make exp directory
    exp_dir, exp_str, exp_id = t_utils.get_exp_dir(conf)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    log.info(f"Running experiment with config:\n{OmegaConf.to_yaml(conf)}\n")
    log.info(f"Saving experiment logs in {exp_dir}")

    if not conf.no_wandb:
        conf_for_wandb = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)
        conf_for_wandb["exp_id"] = exp_id # Add experiment id to conf, for wandb
        # wandb init
        wandb.init(project='trifinger_forward_model', entity='clairec', name=exp_str,
                   config = conf_for_wandb,
                   settings=wandb.Settings(start_method="thread"))

    # Load train and test trajectories for making dataset and for testing MPC
    traj_info = torch.load(conf.demo_path)
    train_trajs = traj_info["train_demos"]
    test_trajs = traj_info["test_demos"]

    # Datasets and dataloaders
    train_dataset = ForwardModelDataset(train_trajs, obj_state_type=conf.algo.obj_state_type, device=device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = ForwardModelDataset(test_trajs, obj_state_type=conf.algo.obj_state_type, device=device)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model dimensions
    if conf.algo.use_ftpos:
        in_dim = train_dataset.a_dim + train_dataset.o_state_dim + train_dataset.ft_state_dim
        out_dim = train_dataset.o_state_dim + train_dataset.ft_state_dim
    else:
        in_dim = train_dataset.a_dim + train_dataset.o_state_dim
        out_dim = train_dataset.o_state_dim
    hidden_dims = list(map(int, str(conf.algo.hidden_dims).split("-"))) # d1-d2 (str) --> [d1, d2] (list of ints)

    model = ForwardModel(in_dim, out_dim, hidden_dims)
    log.info(f"Model:\n{model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.algo.lr)
    loss_fn = torch.nn.MSELoss()

    train(conf, model, loss_fn, optimizer, train_dataloader, test_dataloader, traj_info,
        model_data_dir=exp_dir
    )

if __name__ == '__main__':
    main()

