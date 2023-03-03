#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from setuptools import find_packages
from setuptools import find_namespace_packages


packages = find_packages(where="src") + find_namespace_packages(
    include=["hydra_plugins.*"], where="src"
)
install_requires = [
    "torch >= 1.10.2",
    "torchvision >= 0.11.3",
    "timm==0.6.11",
    "hydra-core",
    "wandb>=0.13",
    "six"
]

setup(
    name="vc_models",
    version="0.1",
    packages=packages,
    package_dir={"": "src"},
    install_requires=install_requires,
    include_package_data=True,
)
