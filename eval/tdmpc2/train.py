import warnings

warnings.filterwarnings("ignore")
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
import torch
import numpy as np
import gym

gym.logger.set_level(40)
import time
from pathlib import Path
from cfg_parse import parse_cfg
from env import make_env, set_seed
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
from termcolor import colored
import logger
import hydra

torch.backends.cudnn.benchmark = True
__CONFIG__, __MODELS__ = "cfgs", "models"


def evaluate(env, agent, cfg, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    episode_successes = []
    for i in range(cfg.eval_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        while not done:
            state = env.state if cfg.get("include_state", False) else None
            action = agent.plan(obs, None, state, eval_mode=True, step=step, t0=t == 0)
            obs, reward, done, info = env.step(action.cpu().numpy())
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        episode_successes.append(info.get("success", 0))
        if video:
            video.save(env_step)
    return np.nanmean(episode_rewards), np.nanmean(episode_successes)


@hydra.main(config_name="default", config_path="config")
def train(cfg: dict):
    """Training script for online TD-MPC."""
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    work_dir = (
        Path(cfg.logging_dir)
        / __MODELS__
        / cfg.task
        / (cfg.get("features", cfg.modality))
        / cfg.algorithm
        / cfg.exp_name
        / str(cfg.seed)
    )
    print(colored("Work dir:", "yellow", attrs=["bold"]), work_dir)
    env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
    print(agent.model)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
        # Collect trajectory
        obs = env.reset()
        state = env.state if cfg.get("include_state", False) else None
        episode = Episode(cfg, obs, state)
        while not episode.done:
            action = agent.plan(obs, None, state, step=step, t0=episode.first)
            obs, reward, done, info = env.step(action.cpu().numpy())
            state = env.state if cfg.get("include_state", False) else None
            episode += (obs, state, action, reward, done)
        if len(episode) < cfg.episode_length:
            print(len(episode), done, episode.reward[: len(episode)].sum().cpu().item())
        assert len(episode) == cfg.episode_length
        buffer += episode

        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = (
                cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            ) // cfg.steps_per_update
            for i in range(num_updates):
                train_metrics.update(
                    agent.update(buffer, step + int(i * cfg.steps_per_update))
                )

        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            "episode": episode_idx,
            "step": step,
            "env_step": env_step,
            "total_time": time.time() - start_time,
            "episode_reward": episode.cumulative_reward,
            "episode_success": info.get("success", 0),
        }
        train_metrics.update(common_metrics)
        L.log(train_metrics, category="train")

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            eval_rew, eval_succ = evaluate(env, agent, cfg, step, env_step, L.video)
            common_metrics.update(
                {"episode_reward": eval_rew, "episode_success": eval_succ}
            )
            L.log(common_metrics, category="eval")
            if cfg.save_model and env_step % cfg.save_freq == 0:
                L.save_model(agent, env_step)

    L.finish()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
