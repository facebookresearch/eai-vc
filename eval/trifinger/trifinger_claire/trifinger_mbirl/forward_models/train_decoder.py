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
import imageio

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))
sys.path.insert(0, os.path.join(base_path, '../..'))

from trifinger_mbirl.forward_models.datasets.decoder_dataset import DecoderDataset
from trifinger_mbirl.forward_models.models.decoder_model import DecoderModel
import utils.data_utils as d_utils
import utils.train_utils as t_utils

# A logger for this file
log = logging.getLogger(__name__)

def train(conf, model, loss_fn, optimizer, train_dataloader, test_dataloader,
          train_dataloader_traj_order,
          model_data_dir=None):

    ckpts_dir = os.path.join(model_data_dir, "ckpts") 
    plots_dir = os.path.join(model_data_dir, "plots") 
    if not os.path.exists(ckpts_dir): os.makedirs(ckpts_dir)
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)

    for epoch in range(conf.algo.n_epochs):
        # Unfreeze network params (params are frozen during mpc rollout)
        for name, param in model.named_parameters():
            param.requires_grad = True

        # Train
        model.train()
        total_train_loss = 0.0
        for i_batch, batch in enumerate(train_dataloader):

            input_tensor = batch["r3m_vec"]
            gt_imgs = batch["rgb_img"]
            optimizer.zero_grad()
            pred_imgs = model(input_tensor)
            loss = loss_fn(pred_imgs, gt_imgs)
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
                input_tensor = batch["r3m_vec"]
                gt_imgs = batch["rgb_img"]
                pred_imgs = model(input_tensor)

                test_loss = loss_fn(pred_imgs, gt_imgs)
                total_test_loss += test_loss.item()
        avg_test_loss = total_test_loss/(i_batch+1)
        print(f"Epoch: {epoch}, test loss: {avg_test_loss}")

        if (epoch+1) % conf.algo.n_epoch_every_save == 0:
            # Save pred images
            for split_name, dataloader in [["train", train_dataloader_traj_order],
                ["test", test_dataloader]]:
                for batch in dataloader:

                    # Make save directory for plots
                    diff = batch["demo_stats"]["diff"][0].detach().item()
                    traj_i = batch["demo_stats"]["id"][0].detach().item()
                    traj_dir = os.path.join(plots_dir, split_name, f"diff-{diff}_traj-{traj_i}")
                    if not os.path.exists(traj_dir): os.makedirs(traj_dir)

                    input_tensor = batch["r3m_vec"]
                    gt_imgs = batch["rgb_img"]
                    pred_imgs = model(input_tensor)
                    model.save_gif(pred_imgs, os.path.join(plots_dir, traj_dir, f'epoch_{epoch+1}.gif'))

        loss_dict = {
                    "train_loss": avg_train_loss, 
                    "test_loss": avg_test_loss, 
                    }

        if not conf.no_wandb:
            t_utils.plot_loss(loss_dict, epoch+1)

        if (epoch+1) % conf.algo.n_epoch_every_save == 0:
            # Save checkpoint
            model_dict = {
                'loss_train'       : loss,
                'model_state_dict' : model.state_dict(),
                'conf'             : conf,
            }
            torch.save(model_dict, f=f'{ckpts_dir}/epoch_{epoch+1}_ckpt.pth')

## Training
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(conf):
    assert conf.algo.name == "decoder", "Need to use algo=decoder when running script"

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
        # wandb init
        conf_for_wandb = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)
        conf_for_wandb["exp_id"] = exp_id # Add experiment id to conf, for wandb
        wandb.init(project='trifinger_decoder', entity='clairec', name=exp_str,
                   config = conf_for_wandb,
                   settings=wandb.Settings(start_method="thread"))

    # Load train and test trajectories for making dataset and for testing MPC
    traj_info = torch.load(conf.demo_path)
    train_trajs = traj_info["train_demos"]
    test_trajs = traj_info["test_demos"]

    # Datasets and dataloaders
    train_dataset = DecoderDataset(train_trajs, traj_info["train_demo_stats"], device=device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    train_dataloader_traj_order = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=False)

    test_dataset = DecoderDataset(test_trajs, traj_info["test_demo_stats"], device=device)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False)

    model = DecoderModel().to(device)
    log.info(f"Model:\n{model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.algo.lr)
    loss_fn = torch.nn.MSELoss()

    #train(conf, model, loss_fn, optimizer, train_dataloader, test_dataloader,
    train(conf, model, loss_fn, optimizer, train_dataloader, test_dataloader,
        train_dataloader_traj_order,
        model_data_dir=exp_dir
    )

if __name__ == '__main__':
    main()