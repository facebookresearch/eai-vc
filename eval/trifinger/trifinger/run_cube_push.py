import os.path as osp
from typing import Dict
import cv2
import gym
import gym.spaces as spaces
import hydra
import torch

# to avoid `GLIBCXX_3.4.30' not found error
import cv2
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from imitation_learning.utils.logging import Logger

from imitation_learning.utils import flatten_info_dict_reader
from imitation_learning.utils.evaluator import CubeEvaluator
from tensordict import TensorDict
from torchrl.envs import (
    ParallelEnv,
    TransformedEnv,
    ToTensorImage,
    Resize,
    Compose,
    ObservationNorm,
    CenterCrop,
)

try:
    from torchrl.envs.utils import step_tensordict as step_tensordict
except ImportError:
    from torchrl.envs.utils import step_mdp as step_tensordict
from torchrl.envs.libs.gym import GymWrapper
from PIL import Image
import trifinger_envs
import torchvision.transforms as T
import wandb
import numpy as np
import random
from rl_utils import common
from pathlib import Path
import argparse
import os
import uuid
import submitit
from trifinger_envs.cube_env import ActionType


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def create_trifinger_env(env_name, seed, env_kwargs, sample_radius, sample_goals):
    # TODO fix gym registration
    if env_name == "ReachEnv-v0":
        env = trifinger_envs.reach_env.ReachEnv(**env_kwargs)
    # hydra param env: new_reach
    elif env_name == "ReachEnv-v1":
        # no logic for random starts here
        env_kwargs["sample_radius"] = sample_radius
        env_kwargs["randomize_all"] = sample_goals
        env = trifinger_envs.new_reach.NewReachEnv(**env_kwargs)
    elif env_name == "CubeReach-v0":
        env_kwargs["sample_radius"] = sample_radius
        env_kwargs["randomize_all"] = sample_goals
        env = trifinger_envs.cube_reach.CubeReachEnv(**env_kwargs)
    elif env_name == "MoveCube-v0":
        # env_kwargs["sample_radius"] = sample_radius
        # env_kwargs["randomize_all"] = sample_goals
        vis_obs = env_kwargs["visual_observation"]
        env_kwargs = {}
        env_kwargs["visual_observation"] = vis_obs
        env_kwargs["random_q_init"] = False
        env_kwargs["action_type"] = ActionType.TORQUE
        env = trifinger_envs.gym_cube_env.MoveCubeEnv(**env_kwargs)
    elif env_name == "NewMoveCube-v0":
        # env_kwargs["sample_radius"] = sample_radius
        # env_kwargs["randomize_all"] = sample_goals
        vis_obs = env_kwargs["visual_observation"]
        env_kwargs = {}
        env_kwargs["visual_observation"] = vis_obs
        env_kwargs["random_q_init"] = False
        env_kwargs["action_type"] = ActionType.TORQUE
        env = trifinger_envs.new_push.NewMoveCubeEnv(**env_kwargs)
    env.seed(seed)
    gym_env = GymWrapper(env, device="cuda:0")
    tensordict = gym_env.reset()
    tensordict = gym_env.rand_step(tensordict)
    return gym_env


class TrifingerEval(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_scaled_success = 0

    def __call__(self, seed=None):
        if seed is not None:
            self.cfg.seed = seed
        cfg = self.cfg

        wandb.require("service")
        self.cfg.device = "cpu" if torch.cuda.device_count() == 0 else "cuda:0"
        print(f"device:{self.cfg.device}")

        device = torch.device(self.cfg.device)

        # TODO replace rl-utils functionality for setting seed:
        #     set_seed(self.cfg.seed)

        set_env_settings = {
            k: hydra_instantiate(v) if isinstance(v, DictConfig) else v
            for k, v in self.cfg.env.env_settings.items()
        }

        # # Get base model, transform, and probing classifier
        if "model" in self.cfg:
            _, embedding_dim, _, _ = hydra.utils.call(self.cfg["model"])

            if self.cfg.camera_id == -1:
                self.cfg.obs_shape = [embedding_dim * 3]
                embedding_size = embedding_dim * 3
            else:
                self.cfg.obs_shape = [embedding_dim]
                embedding_size = embedding_dim
        else:
            embedding_size = 0
            # TODO fix the observation that is returned from reach env

        if "model" in self.cfg:
            self.cfg.visual_observation = True

        env_kwargs = {
            "visual_observation": self.cfg.visual_observation,
            "max_goal_dist": self.cfg.max_goal_dist,
            "camera_id": self.cfg.camera_id,
        }
        dummy_env = create_trifinger_env(
            self.cfg.env.env_name,
            self.cfg.seed,
            env_kwargs,
            self.cfg.start_radius,
            self.cfg.sample_goals,
        )

        reader = flatten_info_dict_reader(self.cfg.info_keys)
        print("Making  " + str(self.cfg.env.env_name))
        obs_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
        obs_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
        parallel_envs = ParallelEnv(
            self.cfg.num_envs,
            lambda: create_trifinger_env(
                self.cfg.env.env_name,
                self.cfg.seed,
                env_kwargs,
                self.cfg.start_radius,
                self.cfg.sample_goals,
            ),
        )
        parallel_envs.to(device)
        envs = TransformedEnv(
            parallel_envs,
            Compose(
                ToTensorImage(),
                Resize(256, 256, "bicubic"),
                CenterCrop(224, 224),
                ObservationNorm(obs_mean, obs_std),
            ),
        )
        # envs.to("cuda:0")
        # envs.set_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        common.core_utils.set_seed(self.cfg.seed)

        steps_per_update = self.cfg.num_steps * self.cfg.num_envs
        num_updates = int(self.cfg.num_env_steps) // steps_per_update

        self.cfg.action_dim = dummy_env.action_space.shape[0]
        self.cfg.action_is_discrete = isinstance(self.cfg.action_dim, spaces.Discrete)

        self.cfg.total_num_updates = num_updates
        print("Instantiating logger....")
        logger: Logger = hydra_instantiate(cfg.logger, full_cfg=self.cfg)
        self.cfg.logger.run_name = logger.run_name
        print("Instantiating policy....")
        policy = hydra_instantiate(self.cfg.policy)
        print("Instantiating updater....")
        updater = hydra_instantiate(
            self.cfg.policy_updater, policy=policy, device=policy.actor.device
        )
        print("Instantiating evaluator....")
        evaluator = hydra_instantiate(
            self.cfg.evaluator,
            envs=envs,
            num_render=1000,  # self.cfg.num_eval_episodes,
            vid_dir=logger.vid_path,
            updater=updater,
            logger=logger,
            device=device,
        )

        start_update = 0
        self.eval_info = {"run_name": logger.run_name}

        if self.cfg.load_checkpoint is not None:
            ckpt = torch.load(self.cfg.load_checkpoint)
            updater.load_state_dict(
                ckpt["updater"], should_load_opt=self.cfg.resume_training
            )
            if self.cfg.load_policy:
                policy.load_state_dict(ckpt["policy"])
            if self.cfg.resume_training:
                start_update = ckpt["update_i"] + 1
            self.eval_info["last_ckpt"] = self.cfg.load_checkpoint

        if self.cfg.only_eval:
            with torch.inference_mode():
                eval_result = evaluator.evaluate(policy, self.cfg.num_eval_episodes, 0)
                logger.collect_infos(eval_result, "eval/", no_rolling_window=True)
                eval_info.update(eval_result)
                logger.interval_log(0, 0)
                logger.close()

            return eval_info

        num_steps = self.cfg.num_steps
        print("Resetting envs ....")
        td = envs.reset()
        max_env_step_idx = (envs.max_episode_len()[0] / envs.step_size()[0]) - 1
        storage_td = TensorDict(
            {}, batch_size=[self.cfg.num_envs, num_steps], device=device
        )
        print("Beginning updates ....")
        env_steps = 0
        for update_i in range(start_update, num_updates):
            is_last_update = update_i == num_updates - 1
            print("rollouts..." + str(update_i))
            for step_idx in range(num_steps):
                with torch.no_grad():
                    policy.act(td)
                envs.step(td)
                storage_td[:, step_idx] = td
                if env_steps >= max_env_step_idx:
                    # scaled_success = compute_success(completed_td,td["total_dist"].detach())
                    logger.collect_info(
                        "train/scaled_success",
                        td["scaled_success"].detach().mean().item(),
                        no_rolling_window=True,
                    )
                    logger.collect_info(
                        "train/scaled_success_reach",
                        td["scaled_success_reach"].detach().mean().item(),
                        no_rolling_window=True,
                    )

                    logger.collect_info("train/epoch", update_i, no_rolling_window=True)
                    scaled_success = td["scaled_success"].detach().mean().item()
                    if scaled_success > self.max_scaled_success:
                        self.max_scaled_success = scaled_success
                    logger.collect_info(
                        "train/max_scaled_success",
                        self.max_scaled_success,
                        no_rolling_window=True,
                    )

                    # success_rate = percentage done and w/ distance less than X / all storage_td
                    for k in self.cfg.info_keys:
                        logger.collect_info(
                            k, td[k].cpu().numpy(), no_rolling_window=True
                        )

                    td.set("reset_workers", td["done"])
                    envs.reset()
                    td["next"]["pixels"] = td["pixels"]
                    logger.collect_env_step_info(td, self.cfg.info_keys)
                    env_steps = -1

                td = step_tensordict(td)
                env_steps += 1

            print("updating policy")
            updater.update(policy, storage_td, logger, envs=envs, device=device)

            if cfg.eval_interval != -1 and (
                update_i % cfg.eval_interval == 0 or is_last_update
            ):
                logger.collect_info(
                    "train/lr",
                    updater.opt.state_dict()["param_groups"][0]["lr"],
                    no_rolling_window=True,
                )
                with torch.inference_mode():
                    eval_result = evaluator.evaluate(
                        policy,
                        cfg.num_eval_episodes,
                        update_i,
                        (update_i % (cfg.eval_interval * 10) == 0),
                    )
                logger.collect_infos(eval_result, "eval/", no_rolling_window=True)
                self.eval_info.update(eval_result)

            if self.cfg.log_interval != -1 and (
                update_i % self.cfg.log_interval == 0 or is_last_update
            ):
                logger.interval_log((update_i + 1), steps_per_update * (update_i + 1))

            if cfg.save_interval != -1 and (
                (update_i + 1) % cfg.save_interval == 0
                or is_last_update
                or update_i == 0
            ):
                save_name = osp.join(logger.save_path, f"ckpt.{update_i}.pth")
                ckpt_name = osp.join(logger.save_path, f"ckpt.pth")
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "updater": updater.state_dict(),
                        "update_i": update_i,
                    },
                    save_name,
                )
                print(f"Saved to {save_name}")
                os.rename(save_name, ckpt_name)
                self.eval_info["last_ckpt"] = ckpt_name

        logger.close()
        return self.eval_info

    def checkpoint(self, *args, **kwargs):
        import os
        import submitit

        print("Entered checkpoint method")
        print(self.cfg)
        self.cfg.load_checkpoint = self.eval_info["last_ckpt"]
        self.cfg.resume_training = True
        # self.cfg.dist_url = get_init_file().as_uri()
        # checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoint.pth")
        print("Requeuing ", self.cfg)
        empty_trainer = type(self)(self.cfg)
        return submitit.helpers.DelayedSubmission(empty_trainer)


@hydra.main(config_path="config", config_name="pretrained")
def main(cfg):
    if cfg.job_dir == "":
        shared_folder = get_shared_folder()
        cfg.job_dir = shared_folder / "%j"

    print("cfg job dir:")
    print(cfg.job_dir)
    executor = submitit.AutoExecutor(folder=cfg.job_dir, slurm_max_num_timeout=10)

    executor.update_parameters(name=cfg.slurm_name)

    num_gpus_per_node = 1
    nodes = 1
    timeout_min = 4300

    partition = "learnfair"
    kwargs = {"slurm_constraint": "volta32gb"}
    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=48,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs,
    )

    executor.update_parameters(slurm_array_parallelism=3)
    s = [1, 2, 3]
    # s = [199, 2431, 378]
    teval = TrifingerEval(cfg)

    jobs = executor.map_array(teval, s)
    print(jobs)

    # job = executor.submit(teval)
    cfg.job_dir = shared_folder / jobs[0].job_id
    cfg.job_id = jobs[0].job_id


if __name__ == "__main__":
    main()
