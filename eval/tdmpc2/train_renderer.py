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
from algorithm.renderer import Renderer
import algorithm.helper as h
from algorithm.helper import RendererBuffer
from dataloader import make_dataset, stack_frames
from termcolor import colored
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import imageio
from logger import make_dir
import pandas as pd
import hydra

os.environ["WANDB_SILENT"] = "true"
import wandb

torch.backends.cudnn.benchmark = True
__LOGS__ = "logs"


def evaluate(env, agent, cfg):
    """Evaluate a trained agent."""
    episode_rewards = []
    for i in range(cfg.eval_episodes):
        if cfg.get("multitask", False):
            env.unwrapped.task_id = i % len(env.unwrapped.tasks)
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        while not done:
            action = agent.plan(
                obs,
                env.unwrapped.task_vec if cfg.get("multitask", False) else None,
                eval_mode=True,
                step=int(1e6),
                t0=t == 0,
            )
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            t += 1
        episode_rewards.append(ep_reward)
    episode_rewards = np.array(episode_rewards)
    return np.nanmean(episode_rewards), episode_rewards


@hydra.main(config_name="default", config_path="config")
def render(cfg: dict):
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
    if os.path.exists(save_dir / "metrics.csv"):
        print("Metrics file already exists. Skipping.")
        return
    env, renderer, buffer, val_buffer = (
        make_env(cfg),
        Renderer(cfg),
        RendererBuffer(cfg),
        RendererBuffer(cfg),
    )

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
    print(renderer.decoder)

    # Load dataset
    assert cfg.get(
        "use_val", False
    ), "Validation dataset is required for simulation experiments"
    dataset = make_dataset(cfg, buffer)
    for episode in dataset._val_episodes:
        val_buffer += episode
    print(f"Buffer contains {buffer.idx} transitions, capacity is {buffer.capacity-1}")
    print(
        f"Validation buffer contains {val_buffer.idx} transitions, capacity is {val_buffer.capacity-1}"
    )
    dataset_summary = dataset.summary
    print(f'\n{colored("Dataset statistics:", "yellow")}\n{dataset_summary}\n')

    # Config
    num_images = 9
    rollout_length = 56

    def eval_encode_decode(buffer, fp=None, num_images=num_images):
        """Evaluate single-step reconstruction error"""
        dictionary = buffer.sample({cfg.modality, cfg.target_modality})
        pred = renderer.encode_decode(dictionary[cfg.modality])
        target = renderer.preprocess_target(dictionary[cfg.target_modality]).cpu()
        if fp is not None and cfg.target_modality == "pixels":
            image_idxs = np.random.randint(len(pred), size=num_images)
            save_image(
                make_grid(
                    torch.cat([pred[image_idxs], target[image_idxs]], dim=0),
                    nrow=num_images,
                ),
                fp,
            )
        return h.mse(pred, target, reduce=True)

    def eval_rollout(
        buffer,
        num_episodes,
        fp=None,
        num_images=num_images,
        rollout_length=rollout_length,
    ):
        start_idx = (
            np.random.randint(cfg.episode_length // 4 - rollout_length - 1)
            + np.random.randint(num_episodes) * cfg.episode_length
        )
        idxs = np.arange(start_idx, start_idx + rollout_length + 1)
        input = buffer.__dict__["_" + cfg.modality][idxs]
        target = buffer.__dict__["_" + cfg.target_modality][idxs]
        action = buffer._action[idxs]
        if cfg.modality == "pixels":
            _input = torch.empty(
                (input.shape[0], input.shape[1] * cfg.frame_stack, *input.shape[2:]),
                dtype=torch.float32,
            )
            input = stack_frames(input, _input, cfg.frame_stack).cuda()
        pred = renderer.imagine(input[0], action)
        target = renderer.preprocess_target(target).cpu()
        mse_rollout = 0
        for t in range(rollout_length):
            mse_rollout += (0.95**t) * h.mse(pred[t], target[t], reduce=True)
        mse_rollout = mse_rollout.item()
        if fp is not None and cfg.target_modality == "pixels":
            image_idxs = np.arange(
                0, rollout_length + 1, rollout_length // (num_images - 1)
            )
            save_image(
                make_grid(
                    torch.cat([pred[image_idxs], target[image_idxs]], dim=0),
                    nrow=num_images,
                ),
                fp,
            )
            imageio.mimsave(
                str(fp).replace(".png", ".mp4"),
                torch.cat([pred, target], dim=-1)
                .mul(255)
                .byte()
                .numpy()
                .transpose(0, 2, 3, 1),
                fps=6,
            )
        return mse_rollout

    # Run training
    metrics = []
    print(colored("Saving to dir:", "yellow"), save_dir)
    for iteration in tqdm(range(cfg.train_iter + 1)):
        # Update model
        common_metrics = renderer.update(buffer)

        if iteration % cfg.eval_freq == 0:
            # Evaluate (training set)
            train_mse = eval_encode_decode(
                buffer, fp=save_dir / f"train_{iteration}.png"
            )
            train_mse_rollout = eval_rollout(
                buffer,
                len(dataset._episodes),
                fp=save_dir / f"train_rollout_{iteration}.png",
            )

            # Evaluate (validation set)
            eval_mse = eval_encode_decode(
                val_buffer, fp=save_dir / f"val_{iteration}.png"
            )
            eval_mse_rollout = eval_rollout(
                val_buffer,
                len(dataset._val_episodes),
                fp=save_dir / f"val_rollout_{iteration}.png",
            )

            # Logging
            metrics.append(
                np.array(
                    [
                        iteration,
                        train_mse,
                        eval_mse,
                        train_mse_rollout,
                        eval_mse_rollout,
                        *common_metrics.values(),
                    ]
                )
            )
            pd.DataFrame(np.array(metrics)).to_csv(
                f"{save_dir}/metrics.csv",
                header=[
                    "iteration",
                    "train_mse",
                    "eval_mse",
                    "train_mse_rollout",
                    "eval_mse_rollout",
                    *common_metrics.keys(),
                ],
                index=None,
            )
            print(
                f"Iteration {iteration}, train mse: {train_mse:.4f}, eval mse: {eval_mse:.4f}, train mse rollout: {train_mse_rollout:.4f}, eval mse rollout: {eval_mse_rollout:.4f}"
            )

            if iteration % cfg.save_freq == 0 and iteration > 0:
                renderer.save(f"{save_dir}/model_{iteration}.pt")


if __name__ == "__main__":
    render()
