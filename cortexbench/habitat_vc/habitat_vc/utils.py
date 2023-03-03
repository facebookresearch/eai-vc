#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import wandb

from habitat.utils.visualizations.utils import (
    images_to_video,
    append_text_to_image,
    draw_collision,
    tile_images,
)
from habitat.utils.visualizations import maps


def setup_wandb(config, train):
    if train:
        file_name = "wandb_id.txt"
        run_name = config.WANDB.name + "_" + str(config.TASK_CONFIG.SEED)
    else:
        ckpt_str = ""
        if ".pth" in config.EVAL_CKPT_PATH_DIR:
            ckpt_str = "_" + config.EVAL_CKPT_PATH_DIR.split("/")[-1].split(".")[1]
        file_name = "wandb_id_eval_" + str(str(config.EVAL.SPLIT)) + ckpt_str + ".txt"
        run_name = config.WANDB.name + "_" + str(config.EVAL.SPLIT)

    wandb_filepath = os.path.join(config.TENSORBOARD_DIR, file_name)

    slurm_info_dict = {
        k[len("SLURM_") :]: v for k, v in os.environ.items() if k.startswith("SLURM_")
    }

    # If file exists, then we are resuming from a previous eval
    if os.path.exists(wandb_filepath):
        with open(wandb_filepath, "r") as file:
            wandb_id = file.read().rstrip("\n")
    else:
        wandb_id = wandb.util.generate_id()
        with open(wandb_filepath, "w") as file:
            file.write(wandb_id)

    wandb.init(
        group=config.WANDB.name,
        job_type=str(config.TASK_CONFIG.SEED),
        id=wandb_id,
        project=config.WANDB.project,
        config={"slurm": slurm_info_dict, **config},
        mode=config.WANDB.mode,
        entity=config.WANDB.entity,
        resume="allow",
        name=run_name,
    )


def poll_checkpoint_folder(
    checkpoint_folder: str,
    previous_ckpt_ind: int,
    suggested_interval: int,
    max_ckpts: int,
) -> Optional[str]:
    r"""Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    checkpoint_folder = glob.escape(checkpoint_folder)
    models_paths = list(filter(os.path.isfile, glob.glob(checkpoint_folder + "/*")))

    models_paths.sort(key=os.path.getmtime)

    if previous_ckpt_ind == -1:
        ind = 0
    else:
        ind = previous_ckpt_ind + suggested_interval

    if ind < len(models_paths):
        return models_paths[ind], ind
    elif ind == max_ckpts and len(models_paths) == max_ckpts:
        return models_paths[-1], len(models_paths) - 1

    return None, previous_ckpt_ind


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    render_obs_images: List[np.ndarray] = []
    for sensor_name in observation:
        if "rgb" in sensor_name:
            rgb = observation[sensor_name]
            if not isinstance(rgb, np.ndarray):
                rgb = rgb.cpu().numpy()

            render_obs_images.append(rgb)
        elif "depth" in sensor_name:
            depth_map = observation[sensor_name].squeeze() * 255.0
            if not isinstance(depth_map, np.ndarray):
                depth_map = depth_map.cpu().numpy()

            depth_map = depth_map.astype(np.uint8)
            depth_map = np.stack([depth_map for _ in range(3)], axis=2)
            render_obs_images.append(depth_map)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation or "imagegoalrotation" in observation:
        if "imagegoal" in observation:
            rgb = observation["imagegoal"]
        else:
            rgb = observation["imagegoalrotation"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        render_obs_images.append(rgb)

    assert len(render_obs_images) > 0, "Expected at least one visual sensor enabled."

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        render_frame = draw_collision(render_frame)

    if "top_down_map" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], render_frame.shape[0]
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)
    return render_frame


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: Union[int, str],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    fps: int = 10,
    verbose: bool = True,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(metric_strs)
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, verbose=verbose)
    if "wandb" in video_option:
        images = np.array(images)
        images = images.transpose(0, 3, 1, 2)
        wandb.log(
            {f"episode{episode_id}_{checkpoint_idx}": wandb.Video(images, fps=fps)}
        )


def add_info_to_image(frame, info):
    string = "d2g: {} | a2g: {} |\nsimple reward: {} |\nsuccess: {} | angle success: {}".format(
        round(info["distance_to_goal"], 3),
        round(info["angle_to_goal"], 3),
        round(info["simple_reward"], 3),
        round(info["success"], 3),
        round(info["angle_success"], 3),
    )
    frame = append_text_to_image(frame, string)
    return frame


def write_json(data, path):
    with open(path, "w") as file:
        file.write(json.dumps(data))


def load_dataset(path):
    with gzip.open(path, "rb") as file:
        data = json.loads(file.read(), encoding="utf-8")
    return data


def load_json_dataset(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data
