"""
This is a launcher script for launching mjrl training using hydra
"""

import os
import time as timer
import hydra
import multiprocessing
from omegaconf import DictConfig, OmegaConf
from train_loop import bc_pvr_train_loop, configure_cluster_GPUs

cwd = os.getcwd()

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_name="BC_config", config_path="config")
def configure_jobs(job_data:dict) -> None:

    print("========================================")
    print("Job Configuration")
    print("========================================")

    job_data = OmegaConf.structured(OmegaConf.to_yaml(job_data))

    # configure GPUs
    # os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    physical_gpu_id = configure_cluster_GPUs(job_data['env_kwargs']['render_gpu_id'])
    job_data['env_kwargs']['render_gpu_id'] = physical_gpu_id

    job_data['cwd'] = cwd
    with open('job_config.json', 'w') as fp:
        OmegaConf.save(config=job_data, f=fp.name)
    print(OmegaConf.to_yaml(job_data))
    bc_pvr_train_loop(job_data)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    configure_jobs()