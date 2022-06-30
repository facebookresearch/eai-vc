"""
This is a launcher script for launching mjrl training using hydra
"""

import os
import hydra
import multiprocessing
from omegaconf import DictConfig, OmegaConf

cwd = os.getcwd()

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_path="config", config_name="BC_config", version_base="1.1")
def configure_jobs(job_data: dict) -> None:

    print("========================================")
    print("Job Configuration")
    print("========================================")

    job_data = OmegaConf.structured(OmegaConf.to_yaml(job_data))

    from rep_eval.visual_il.train_loop import bc_pvr_train_loop, configure_cluster_GPUs
    import wandb

    # configure GPUs
    # os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    physical_gpu_id = configure_cluster_GPUs(job_data['env_kwargs']['render_gpu_id'])
    job_data['env_kwargs']['render_gpu_id'] = physical_gpu_id

    job_data['cwd'] = cwd
    with open('job_config.json', 'w') as fp:
        OmegaConf.save(config=job_data, f=fp.name)
    # print(OmegaConf.to_yaml(job_data))
    print("Arch : %s" % job_data['embedding'])

    wandb_run = None
    if job_data['wandb_logging'] and 'WANDB_USER' in os.environ:
        wandb_run = wandb.init(project=job_data['wandb_project'], entity=os.environ['WANDB_USER'], 
                               config=OmegaConf.to_container(job_data, resolve=True))

    bc_pvr_train_loop(job_data, wandb_run)
    wandb.finish()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    configure_jobs()