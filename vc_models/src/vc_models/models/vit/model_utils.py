#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import urllib

import vc_models
import hydra
import omegaconf
import six

VC1_BASE_NAME = "vc1_vitb"
VC1_LARGE_NAME = "vc1_vitl"
_EAI_VC1_BASE_URL = "https://dl.fbaipublicfiles.com/eai-vc/"


# progress_bar and download_url from 
# https://github.com/facebookresearch/Detectron/blob/1809dd41c1ffc881c0d6b1c16ea38d08894f8b6d/detectron/utils/io.py
def _progress_bar(count, total):
    """Report download progress.
    Credit:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(
        '  [{}] {}% of {:.1f}MB file  \r'.
        format(bar, percents, total / 1024 / 1024)
    )
    sys.stdout.flush()
    if count >= total:
        sys.stdout.write('\n')

def _download_url(
    url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar
):
    """Download url and write it to dst_file_path.
    Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    try:
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        print(f"Error downloading model from {_EAI_VC1_BASE_URL}:\n{e}")
        raise
    if six.PY2:
        total_size = response.info().getheader('Content-Length').strip()
    else:
        total_size = response.info().get('Content-Length').strip()
    total_size = int(total_size)
    bytes_so_far = 0

    with open(dst_file_path, 'wb') as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if progress_hook:
                progress_hook(bytes_so_far, total_size)
            f.write(chunk)

    return bytes_so_far

def download_model_if_needed(ckpt_file):
    model_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..')
    ckpt_file = os.path.join(model_base_dir,ckpt_file)
    if not os.path.isfile(ckpt_file):
        os.makedirs(os.path.dirname(ckpt_file),exist_ok=True)

        model_name = ckpt_file.split("/")[-1]
        model_url = _EAI_VC1_BASE_URL + model_name
        _download_url(model_url,ckpt_file)


def load_model(model_name):
    """
    Loads a model from the vc_models package.
    Args:
        model_name (str): name of the model to load
    Returns:
        model (torch.nn.Module): the model
        embedding_dim (int): the dimension of the embedding
        transform (torchvision.transforms): the transform to apply to the image
        metadata (dict): the metadata of the model
    """
    models_filepath = os.path.dirname(os.path.abspath(vc_models.__file__))
    
    cfg_path = os.path.join(models_filepath,"conf", "model", f"{model_name}.yaml")
    
    model_cfg = omegaconf.OmegaConf.load(cfg_path)
    # returns tuple of model, embedding_dim, transform, metadata
    return hydra.utils.call(model_cfg)

