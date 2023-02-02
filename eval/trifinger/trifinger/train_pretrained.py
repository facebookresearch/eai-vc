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
import trifinger_envs
from PIL import Image
import trifinger_envs
import torchvision.transforms as T
import wandb
import numpy as np
import random
from rl_utils import common


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
        env = trifinger_envs.gym_cube_env.MoveCubeEnv(**env_kwargs)
    env.seed(seed)
    gym_env = GymWrapper(env, device="cuda:0")
    tensordict = gym_env.reset()
    tensordict = gym_env.rand_step(tensordict)
    return gym_env


@hydra.main(config_path="config", config_name="pretrained")
def main(cfg) -> Dict[str, float]:
    wandb.require("service")
    cfg.device = "cpu" if torch.cuda.device_count() == 0 else "cuda:0"
    print(f"device:{cfg.device}")

    device = torch.device(cfg.device)

    # TODO replace rl-utils functionality for setting seed:
    #     set_seed(cfg.seed)

    set_env_settings = {
        k: hydra_instantiate(v) if isinstance(v, DictConfig) else v
        for k, v in cfg.env.env_settings.items()
    }

    # # Get base model, transform, and probing classifier
    if "model" in cfg:
        _, embedding_dim, _, _ = hydra.utils.call(cfg["model"])

        if cfg.camera_id == -1:
            cfg.obs_shape = [embedding_dim * 3]
            embedding_size = embedding_dim * 3
        else:
            cfg.obs_shape = [embedding_dim]
            embedding_size = embedding_dim
    else:
        embedding_size = 0
        # TODO fix the observation that is returned from reach env

    if "model" in cfg:
        cfg.visual_observation = True

    env_kwargs = {
        "visual_observation": cfg.visual_observation,
        "max_goal_dist": cfg.max_goal_dist,
        "camera_id": cfg.camera_id,
    }
    dummy_env = create_trifinger_env(
        cfg.env.env_name, cfg.seed, env_kwargs, cfg.start_radius, cfg.sample_goals
    )

    reader = flatten_info_dict_reader(cfg.info_keys)
    print("Making  " + str(cfg.env.env_name))
    obs_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
    obs_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
    parallel_envs = ParallelEnv(
        cfg.num_envs,
        lambda: create_trifinger_env(
            cfg.env.env_name,
            cfg.seed,
            env_kwargs,
            cfg.start_radius,
            cfg.sample_goals,
        ),
    )
    parallel_envs.to(device)
    envs = TransformedEnv(
        parallel_envs,
        Compose(
            ToTensorImage(),
            Resize(256, 256),
            CenterCrop(224, 224),
            ObservationNorm(obs_mean, obs_std),
        ),
    )
    # envs.to("cuda:0")
    # envs.set_seed(cfg.seed)
    random.seed(cfg.seed)
    common.core_utils.set_seed(cfg.seed)

    steps_per_update = cfg.num_steps * cfg.num_envs
    num_updates = int(cfg.num_env_steps) // steps_per_update

    cfg.action_dim = dummy_env.action_space.shape[0]
    cfg.action_is_discrete = isinstance(cfg.action_dim, spaces.Discrete)

    cfg.total_num_updates = num_updates
    print("Instantiating logger....")
    logger: Logger = hydra_instantiate(cfg.logger, full_cfg=cfg)
    print("Instantiating policy....")
    policy = hydra_instantiate(cfg.policy)
    print("Instantiating updater....")
    updater = hydra_instantiate(
        cfg.policy_updater, policy=policy, device=policy.actor.device
    )
    print("Instantiating evaluator....")
    evaluator: CubeEvaluator = hydra_instantiate(
        cfg.evaluator,
        envs=envs,
        num_render=cfg.num_eval_episodes,
        vid_dir=logger.vid_path,
        updater=updater,
        logger=logger,
        device=device,
    )

    start_update = 0
    if cfg.load_checkpoint is not None:
        ckpt = torch.load(cfg.load_checkpoint)
        updater.load_state_dict(ckpt["updater"], should_load_opt=cfg.resume_training)
        if cfg.load_policy:
            policy.load_state_dict(ckpt["policy"])
        if cfg.resume_training:
            start_update = ckpt["update_i"] + 1

    eval_info = {"run_name": logger.run_name}

    if cfg.only_eval:
        with torch.inference_mode():
            eval_result = evaluator.evaluate(policy, cfg.num_eval_episodes, 0)
            logger.collect_infos(eval_result, "eval.", no_rolling_window=True)
            eval_info.update(eval_result)
            logger.interval_log(0, 0)
            logger.close()

        return eval_info

    num_steps = cfg.num_steps
    print("Resetting envs ....")
    td = envs.reset()
    max_env_step_idx = envs.max_episode_len()[0] / envs.step_size()[0] - 1

    storage_td = TensorDict({}, batch_size=[cfg.num_envs, num_steps], device=device)
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
                    "scaled_success",
                    td["scaled_success"].detach().mean().item(),
                    no_rolling_window=True,
                )
                # success_rate = percentage done and w/ distance less than X / all storage_td

                # logger.collect_env_step_info(td, cfg.info_keys)
                for k in cfg.info_keys:
                    logger.collect_info(k, td[k].cpu().numpy(), no_rolling_window=True)

                td.set("reset_workers", td["done"])
                envs.reset()
                td["next"]["pixels"] = td["pixels"]
                logger.collect_env_step_info(td, cfg.info_keys)
                env_steps = -1

            td = step_tensordict(td)
            env_steps += 1

        print("updating policy")
        updater.update(policy, storage_td, logger, envs=envs, device=device)

        if cfg.eval_interval != -1 and (
            update_i % cfg.eval_interval == 0 or is_last_update
        ):
            with torch.inference_mode():
                eval_result = evaluator.evaluate(
                    policy, cfg.num_eval_episodes, update_i
                )
            logger.collect_infos(eval_result, "eval.", no_rolling_window=True)
            eval_info.update(eval_result)

        if cfg.log_interval != -1 and (
            update_i % cfg.log_interval == 0 or is_last_update
        ):
            logger.interval_log((update_i + 1), steps_per_update * (update_i + 1))

        if cfg.save_interval != -1 and (
            (update_i + 1) % cfg.save_interval == 0 or is_last_update
        ):
            save_name = osp.join(logger.save_path, f"ckpt.{update_i}.pth")
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "updater": updater.state_dict(),
                    "update_i": update_i,
                },
                save_name,
            )
            print(f"Saved to {save_name}")
            eval_info["last_ckpt"] = save_name

    logger.close()
    return eval_info


if __name__ == "__main__":
    main()
