import glob
import os
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import torch
from timm.models.vision_transformer import resize_pos_embed
import wandb

from habitat.utils.visualizations.utils import images_to_video

from eai.models.resnet_gn import ResNet
from eai.models.vit import VisionTransformer


def load_encoder(encoder, path):
    assert os.path.exists(path)
    if isinstance(encoder.backbone, ResNet):
        state_dict = torch.load(path, map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return encoder.load_state_dict(state_dict=state_dict, strict=False)
    elif isinstance(encoder.backbone, VisionTransformer):
        model = encoder.backbone
        state_dict = torch.load(path, map_location="cpu")["model"]
        if state_dict["pos_embed"].shape != model.pos_embed.shape:
            state_dict["pos_embed"] = resize_pos_embed(
                state_dict["pos_embed"],
                model.pos_embed,
                getattr(model, "num_tokens", 1),
                model.patch_embed.grid_size,
            )
        return model.load_state_dict(state_dict=state_dict, strict=False)
    else:
        raise ValueError("unknown encoder backbone")

def setup_wandb(config, train):
    if train:
        file_name = "wandb_id.txt"
        project_name = "imagenav_training"
        run_name = config.WANDB_NAME + "_" + str(config.TASK_CONFIG.SEED)
    else:
        file_name = "wandb_id_eval_" + str(config.EVAL.SPLIT) + ".txt"
        project_name = "imagenav_testing"
        run_name = config.WANDB_NAME + "_" + str(config.EVAL.SPLIT) + "_" + \
            str(config.TASK_CONFIG.SEED)

    wandb_filepath = os.path.join(config.TENSORBOARD_DIR, file_name)

    # If file exists, then we are resuming from a previous eval
    if os.path.exists(wandb_filepath):
        with open(wandb_filepath, 'r') as file:
            wandb_id = file.read().rstrip('\n')
        
        wandb.init(
            group=config.WANDB_NAME,
            job_type=str(config.TASK_CONFIG.SEED),
            id=wandb_id,
            project=project_name,
            config=config,
            mode=config.WANDB_MODE,
            resume='allow'
        )
    
    else:
        wandb_id=wandb.util.generate_id()
        
        with open(wandb_filepath, 'w') as file:
            file.write(wandb_id)
        
        wandb.init(
            group=config.WANDB_NAME,
            job_type=str(config.TASK_CONFIG.SEED),
            id=wandb_id,
            project=project_name,
            config=config,
            mode=config.WANDB_MODE
        )

    wandb.run.name = run_name
    wandb.run.save()

def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int, suggested_interval: int
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
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )

    models_paths.sort(key=os.path.getmtime)

    if previous_ckpt_ind == -1:
        ind = 0
    else:
        ind = previous_ckpt_ind + suggested_interval

    if ind < len(models_paths):
        return models_paths[ind], ind
    elif previous_ckpt_ind + 1 < len(models_paths):
        return models_paths[-1], len(models_paths) - 1

    return None, previous_ckpt_ind

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

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, verbose=verbose)
    if "wandb" in video_option:
        images = images.transpose(0, 3, 1, 2)
        wandb.log({f"episode{episode_id}_{checkpoint_idx}": wandb.Video(images, fps=fps)})

