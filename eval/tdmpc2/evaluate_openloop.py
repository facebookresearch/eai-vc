import warnings

warnings.filterwarnings("ignore")
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
import torch
import numpy as np
import gym

gym.logger.set_level(40)
from pathlib import Path
from cfg_parse import parse_cfg
from env import make_env, set_seed
from algorithm.tdmpc import TDMPC
from algorithm.renderer import Renderer
from termcolor import colored
import imageio
from logger import make_dir
import pandas as pd
import hydra

os.environ["WANDB_SILENT"] = "true"
import wandb

torch.backends.cudnn.benchmark = True
__LOGS__ = "logs"


@hydra.main(config_name="default", config_path="config")
def open_loop(cfg: dict):
    """Rendering script for evaluating learned representations."""
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    save_dir = make_dir(
        Path(cfg.logging_dir)
        / "renderer"
        / cfg.task
        / (cfg.features if cfg.modality == "features" else cfg.modality)
        / cfg.target_modality
        / cfg.exp_name
        / str(cfg.seed)
    )
    if not os.path.exists(save_dir / "model_10000.pt"):
        print("Failed to find renderer model. Please train the renderer first.")
        return
    env, renderer = make_env(cfg), Renderer(cfg)

    # Load agent from wandb
    run_name = "renderer" + str(np.random.randint(0, int(1e6)))
    run = wandb.init(
        job_type="renderer",
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=run_name,
        tags="renderer",
    )
    agent = TDMPC(cfg)
    tdmpc_artifacts = [
        f"{cfg.task}-{cfg.modality}-{cfg.exp_name}-{cfg.seed}-chkpt:v{i}"
        for i in reversed(range(10))
    ]
    artifact = None
    for tdmpc_artifact in tdmpc_artifacts:
        try:
            artifact = run.use_artifact(tdmpc_artifact, type="model")
            print(f"Loading TDMPC artifact {tdmpc_artifact}")
            break
        except:
            continue
    if artifact is None:
        print("No TDMPC artifact found. Skipping.")
        return
    artifact_dir = Path(artifact.download())
    agent.load(artifact_dir / os.listdir(artifact_dir)[0])
    renderer.set_tdmpc_agent(agent)
    renderer.load(save_dir / "model_10000.pt")
    print(renderer.decoder)

    # Evaluate
    print(colored("Evaluating open loop...", "yellow"))
    openloop_dir = make_dir(
        Path(cfg.logging_dir)
        / "openloop"
        / cfg.task
        / (cfg.features if cfg.modality == "features" else cfg.modality)
        / cfg.exp_name
        / str(cfg.horizon)
        / str(cfg.seed)
    )
    metrics, episode_rewards = [], []
    for i in range(cfg.eval_episodes):
        obs, done, ep_reward = env.reset(), False, 0
        pred_frames, gt_frames = [], []
        while not done:
            actions = agent.plan(
                obs, None, eval_mode=True, step=int(1e6), t0=True, open_loop=True
            )
            _pred_frames = renderer.imagine(torch.from_numpy(obs), actions).permute(
                0, 2, 3, 1
            )
            for t, action in enumerate(actions.cpu().numpy()):
                pred_frames.append(_pred_frames[t])
                gt_frames.append(env.render(height=64, width=64))
                obs, reward, done, _ = env.step(action)
                ep_reward += reward
                if done:
                    break
        print("Reward:", ep_reward)
        pred_frames = torch.stack(pred_frames).mul(255).byte()
        gt_frames = torch.from_numpy(np.array(gt_frames))
        imageio.mimsave(
            openloop_dir / f"h{cfg.horizon}_{i}.gif",
            torch.cat((pred_frames, gt_frames), dim=2),
        )
        metrics.append(
            np.array([i, cfg.modality, cfg.exp_name, cfg.horizon, cfg.seed, ep_reward])
        )
        pd.DataFrame(np.array(metrics)).to_csv(
            f"{openloop_dir}/metrics.csv",
            header=[
                "episode",
                "modality",
                "exp_name",
                "horizon",
                "seed",
                "episode_reward",
            ],
            index=None,
        )
        episode_rewards.append(ep_reward)

    episode_rewards = np.array(episode_rewards)
    print("Mean reward:", episode_rewards.mean())


if __name__ == "__main__":
    open_loop()
