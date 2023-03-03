#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import omegaconf
import hydra
import vc_models

vc_models_abs_path = os.path.dirname(os.path.abspath(vc_models.__file__))

MODEL_NAMES = vc_models.vc_model_zoo



# # assumes directory contains nested directories or .yaml model files
# def find_models(root_path):
#     models = {}
#     for f in os.listdir(root_path):
#         if os.path.isdir(os.path.join(root_path, f)):
#             temp_d = find_models(os.path.join(root_path, f))
#             temp_d.update(models)
#             models = temp_d
#         elif f.endswith(".yaml"):
#             models[f.split(".")[0]] = os.path.join(root_path, f)
#     return models


# VC_MODEL_NAMES = find_models(
#     os.path.join(vc_models.vc_models_dir_path, "conf/model")
# )

# def get_model_and_transform(model_name, device="cpu"):
#     ## Pretrained VC models
#     if model_name not in MODEL_NAMES:
#         raise NameError("Invalid model_name")
#     return get_vc_model_and_transform(
#         model_name, device=device
#     )  

def get_vc_model_and_transform(model_name, device="cpu", use_compression_layer=False):
    if model_name not in MODEL_NAMES:
         raise NameError("Invalid vc model name")
    # Assumes models are in top level of vc/conf/model
    cfg_path = os.path.join(vc_models_abs_path, "conf", "model", f"{model_name}.yaml")
    main_model_cfg = omegaconf.OmegaConf.load(cfg_path)

    if use_compression_layer:
        if "model" in main_model_cfg.model:
            model_cfg = main_model_cfg.model.model
        else:
            model_cfg = main_model_cfg.model
        model_cfg.global_pool = not use_compression_layer
        model_cfg.use_cls = not use_compression_layer

    model, embedding_dim, transform, metadata = hydra.utils.call(main_model_cfg)
       
    return model, transform, embedding_dim

