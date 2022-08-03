import glob
import os
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import cv2
import numpy as np
from scipy import interpolate
import torch
from timm.models.vision_transformer import resize_pos_embed
import wandb

from habitat.utils.visualizations.utils import (
    images_to_video,
    append_text_to_image,
    draw_collision,
    tile_images,
)
from habitat.utils.visualizations import maps

from eai.models.resnet_gn import ResNet
from eai.models.vit import VisionTransformer
from eai.models.beit import Beit

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
    elif isinstance(encoder.backbone, Beit):
        model = encoder.backbone
        state_dict = torch.load(path, map_location="cpu")["model"]
        return load_data2vec(state_dict, model)
    else:
        raise ValueError("unknown encoder backbone")

def load_data2vec(checkpoint_model, model):
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    if model.use_rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        print("Expand the shared relative position embedding to each transformer block. ")
        num_layers = model.get_num_layers()
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.grid_size
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                print("Position interpolate for %s from %dx%d to %dx%d" % (
                    key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                print("Original positions = %s" % str(x))
                print("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias

    # interpolate position embedding
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

    return model.load_state_dict(checkpoint_model, strict=False)

def setup_wandb(config, train, project_name="imagenav"):
    if train:
        file_name = "wandb_id.txt"
        project_name = project_name + "_training"
        run_name = config.WANDB_NAME + "_" + str(config.TASK_CONFIG.SEED)
    else:
        file_name = "wandb_id_eval_" + str(config.EVAL.SPLIT) + ".txt"
        project_name = project_name + "_testing"
        ckpt_str = "_"
        if os.path.isfile(config.EVAL_CKPT_PATH_DIR):
            ckpt_str = "_" + config.EVAL_CKPT_PATH_DIR.split("/")[-1].split(".")[1] + "_"
        run_name = config.WANDB_NAME + "_" + str(config.EVAL.SPLIT) + ckpt_str + \
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
    checkpoint_folder: str, previous_ckpt_ind: int, suggested_interval: int, max_ckpts: int
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

    assert (
        len(render_obs_images) > 0
    ), "Expected at least one visual sensor enabled."

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

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, verbose=verbose)
    if "wandb" in video_option:
        images = np.array(images)
        images = images.transpose(0, 3, 1, 2)
        wandb.log({f"episode{episode_id}_{checkpoint_idx}": wandb.Video(images, fps=fps)})

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
