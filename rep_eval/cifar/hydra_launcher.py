"""
This is a launcher script for launching CIFAR-10 linear probing using hydra
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf

cwd = os.getcwd()

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_path="config", config_name="cifar_lin_probe", version_base="1.1")
def configure_jobs(job_data: dict) -> None:

    print("========================================")
    print("Job Configuration")
    print("========================================")

    job_data = OmegaConf.structured(OmegaConf.to_yaml(job_data))

    import torch
    import wandb
    from rep_eval.utils.model_loading import load_pvr_model, MODEL_LIST
    from rep_eval.cifar.eval_model_cifar import probe_model_eval
    
    assert job_data['model'] in MODEL_LIST
    
    print("-------------------------")
    print("Arch : %s" % job_data['model'])
    
    wandb_run = None
    if job_data['wandb_logging'] and 'WANDB_USER' in os.environ:
        wandb_run = wandb.init(project=job_data['wandb_project'], entity=os.environ['WANDB_USER'], 
                               config=OmegaConf.to_container(job_data, resolve=True))
    
    # Train the probe
    probe_model_eval(job_data, wandb_run)
    wandb.finish()
    
if __name__ == "__main__":
    configure_jobs()
