import os, time as timer, hydra
from omegaconf import DictConfig, OmegaConf

cwd = os.getcwd()

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_name="cifar_lin_probe", config_path="config")
def configure_jobs(job_data:dict) -> None:

    print("========================================")
    print("Job Configuration")
    print("========================================")

    job_data = OmegaConf.structured(OmegaConf.to_yaml(job_data))

    import torch, torchvision, torchvision.transforms as T
    import numpy as np, wandb
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from model_loading import load_pvr_model, MODEL_LIST
    from eval_model_cifar import probe_model_eval, ClassifierModel
    
    assert job_data['model'] in MODEL_LIST
    
    print("-------------------------")
    print("Arch : %s" % job_data['model'])
    
    wandb_run = wandb.init(project=job_data['wandb_project'], entity=job_data['wandb_user'], 
                           config=OmegaConf.to_container(job_data, resolve=True))

    # Get base model, transform, and probing classifier
    model, embedding_dim, transform = load_pvr_model(job_data['model'])
    linear_probe = torch.nn.Sequential(
                                torch.nn.BatchNorm1d(embedding_dim),
                                torch.nn.Linear(embedding_dim, 10),
                               )
    # Train the probe
    probe_model_eval(job_data, model, transform, linear_probe, embedding_dim, wandb_run)
    wandb.finish()
    
if __name__ == "__main__":
    configure_jobs()
