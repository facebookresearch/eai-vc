#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import vc_models
import hydra
import omegaconf
import requests
import os

from tqdm import tqdm

from os.path import expanduser

VC1_BASE_NAME = "vc1_vitb"
VC1_LARGE_NAME = "vc1_vitl"

_EAI_VC1_BASE_REPO = "facebook/vc1-base"
_EAI_VC1_LARGE_REPO = "facebook/vc1-large"

HG_BASE_URL = "https://huggingface.co/{model_repo}/resolve/main/pytorch_model.bin"

CORTEX_DIR = os.environ.get("CORTEX_DIR", os.path.join(expanduser("~"), ".cortex"))

VC_MODELS_DIR_PATH = os.path.dirname(os.path.abspath(vc_models.__file__))
VC_MODELS_CONFIG_SUBDIR = "conf/model"
VC_MODELS_CONFIG_DIR_PATH = os.path.join(VC_MODELS_DIR_PATH, VC_MODELS_CONFIG_SUBDIR)
VC_MODEL_CONFIG_EXT = ".yaml"


def _get_vc_models_config_files():
    return [
        f
        for f in os.listdir(VC_MODELS_CONFIG_DIR_PATH)
        if f.endswith(VC_MODEL_CONFIG_EXT)
    ]


def _extract_model_name_from_config(config_filename):
    return config_filename.split(".")[0]


vc_models_config_files = _get_vc_models_config_files()
vc_model_zoo = [_extract_model_name_from_config(f) for f in vc_models_config_files]


def download_file(url, save_path, chunk_size=128):
    """
    Downloads a file from a url to a local path.
    Args:
        url (str): url to download from
        save_path (str): path to save the file to
        chunk_size (int): size of chunks to download
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(save_path, "wb") as file:
        for data in tqdm(
            response.iter_content(chunk_size), total=total_size // chunk_size, unit="KB"
        ):
            file.write(data)


def download_model_if_needed(ckpt_file):
    """
    Downloads a model from the vc_models package if it is not already downloaded.
    Args:
        ckpt_file (str): path to the checkpoint file
    Returns:
        ckpt_path (str): absolute path to the checkpoint file

    """
    if not os.path.isabs(ckpt_file):
        ckpt_path = os.path.join(CORTEX_DIR, ckpt_file)
    else:
        ckpt_path = ckpt_file

    if not os.path.isfile(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        if model_name == VC1_BASE_NAME:
            repo_name = _EAI_VC1_BASE_REPO
        elif model_name == VC1_LARGE_NAME:
            repo_name = _EAI_VC1_LARGE_REPO
        else:
            raise ValueError(
                f"""The the file {ckpt_path} was not found and there is no url to be downloaded from. 
                        Please, fix the path in the model config."""
            )

        print(f"Downloading {model_name} to {ckpt_path}.")
        download_file(url=HG_BASE_URL.format(model_repo=repo_name), save_path=ckpt_path)
        print(f"Downloaded file {ckpt_path}.")

        # Check the size of the file if it less than 100MB, then raise an error
        if os.path.getsize(ckpt_path) < 100 * 10 ^ 6:
            with open(ckpt_path, "r") as f:
                file_content = f.read()
            raise ValueError(
                f"""The download of the file {ckpt_path} was unsuccessful. 
                    The file is empty or too small. The file content is: {file_content}"""
            )

    return ckpt_path


def load_model(model_name):
    """
    Loads a model from the vc_models package.
    Args:
        model_name (str): name of the model to load like "vc1_vitb"
    Returns:
        model (torch.nn.Module): the model
        embedding_dim (int): the dimension of the embedding
        transform (torchvision.transforms): the transform to apply to the image
        metadata (dict): the metadata of the model
    """
    cfg_path = os.path.join(VC_MODELS_CONFIG_DIR_PATH, f"{model_name}.yaml")

    model_cfg = omegaconf.OmegaConf.load(cfg_path)
    # returns tuple of model, embedding_dim, transform, metadata
    return hydra.utils.call(model_cfg)
