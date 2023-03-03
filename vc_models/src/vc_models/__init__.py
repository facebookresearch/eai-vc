#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
vc_models_dir_path = os.path.dirname(os.path.abspath(__file__))
vc_models_config_files = os.listdir(vc_models_dir_path + "/conf/model")
vc_model_zoo = [
    f.split(".")[0] for f in vc_models_config_files if f.endswith(".yaml")
]
